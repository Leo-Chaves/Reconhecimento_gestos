"""Theremin Gestual — Instrumento musical via webcam com controle dinâmico de volume.

Reconhece gestos geométricos da mão via MediaPipe Hands para tocar notas da
escala de Dó Maior e modula a intensidade sonora continuamente conforme a altura
vertical do punho na imagem da câmera.
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np

import audio_engine

# Parâmetros de mapeamento da altura do punho para volume
Y_TOPO: float = 0.15
Y_BASE: float = 0.85
VOL_MIN: float = 0.10
VOL_MAX: float = 1.00
EMA_PESO_ATUAL: float = 0.70
EMA_PESO_BRUTO: float = 0.30

# Parâmetros de detecção de gestos e debounce
LIMIAR_PINCA: float = 0.05
FRAMES_CONFIRMACAO: int = 3

# Índices dos landmarks (MediaPipe Hands)
PUNHO = 0
POLEGAR_MCP = 2
POLEGAR_IP = 3
POLEGAR_TIP = 4
INDICADOR_PIP = 6
INDICADOR_TIP = 8
MEDIO_PIP = 10
MEDIO_TIP = 12
ANELAR_PIP = 14
ANELAR_TIP = 16
MINDINHO_MCP = 17
MINDINHO_PIP = 18
MINDINHO_TIP = 20

PARES_DEDOS: tuple[tuple[int, int], ...] = (
    (INDICADOR_TIP, INDICADOR_PIP),
    (MEDIO_TIP, MEDIO_PIP),
    (ANELAR_TIP, ANELAR_PIP),
    (MINDINHO_TIP, MINDINHO_PIP),
)

MAPA_NOTAS_DEDOS: dict[tuple[bool, bool, bool, bool], str | None] = {
    (True, False, False, False): "DO",
    (True, True, False, False): "RE",
    (True, True, True, False): "MI",
    (True, False, False, True): "SI",
    (False, False, False, False): None,
}


def calcular_volume_por_altura(y_punho: float, vol_suave_anterior: float) -> float:
    """Mapeia ``landmarks[0].y`` para ``[0.10, 1.00]`` com suavização EMA."""
    y_limitado = max(Y_TOPO, min(Y_BASE, float(y_punho)))
    fator = (Y_BASE - y_limitado) / (Y_BASE - Y_TOPO)
    vol_bruto = VOL_MIN + fator * (VOL_MAX - VOL_MIN)
    vol_suave = EMA_PESO_ATUAL * vol_suave_anterior + EMA_PESO_BRUTO * vol_bruto
    return max(VOL_MIN, min(VOL_MAX, vol_suave))


def _distancia_2d(lm_a: object, lm_b: object) -> float:
    """Calcula a distância euclidiana normalizada 2D entre dois landmarks."""
    return math.hypot(lm_a.x - lm_b.x, lm_a.y - lm_b.y)


def _polegar_aberto(landmarks: Sequence[object]) -> bool:
    """Verifica se o polegar está estendido para fora da palma (ambidestro)."""
    dist_ponta_mindinho = _distancia_2d(landmarks[POLEGAR_TIP], landmarks[MINDINHO_MCP])
    dist_articulacao_mindinho = _distancia_2d(landmarks[POLEGAR_IP], landmarks[MINDINHO_MCP])
    dist_base_mindinho = _distancia_2d(landmarks[POLEGAR_MCP], landmarks[MINDINHO_MCP])
    return (
        dist_ponta_mindinho > dist_articulacao_mindinho
        and dist_ponta_mindinho > dist_base_mindinho
    )


def classificar_gesto(landmarks: Sequence[object]) -> tuple[bool, str | None]:
    """Classifica os 21 landmarks em uma nota musical ou silêncio.

    Returns:
        Tupla ``(valido, nota)``, onde ``(True, None)`` representa punho fechado
        (silêncio intencional) e ``(False, None)`` representa postura intermediária
        não mapeada (ignorada pelo debounce).
    """
    if _distancia_2d(landmarks[POLEGAR_TIP], landmarks[INDICADOR_TIP]) < LIMIAR_PINCA:
        return True, "LA"

    dedos = tuple(
        landmarks[tip].y < landmarks[pip].y
        for tip, pip in PARES_DEDOS
    )

    if dedos == (True, True, True, True):
        return True, ("SOL" if _polegar_aberto(landmarks) else "FA")

    if dedos in MAPA_NOTAS_DEDOS:
        return True, MAPA_NOTAS_DEDOS[dedos]

    return False, None


def _cor_vu_meter(nivel: float) -> tuple[int, int, int]:
    """Interpola cor BGR de verde -> amarelo -> vermelho conforme o nível [0.1, 1.0]."""
    progresso = max(0.0, min(1.0, (nivel - VOL_MIN) / (VOL_MAX - VOL_MIN)))
    if progresso < 0.5:
        t = progresso * 2.0
        # Verde (60, 220, 60) -> Amarelo (0, 225, 255)
        b = int(60 * (1.0 - t))
        g = int(220 + 5 * t)
        r = int(60 + 195 * t)
    else:
        t = (progresso - 0.5) * 2.0
        # Amarelo (0, 225, 255) -> Vermelho (50, 60, 245)
        b = int(50 * t)
        g = int(225 * (1.0 - t) + 60 * t)
        r = int(255 - 10 * t)
    return (b, g, r)


def desenhar_hud(
    frame: np.ndarray,
    nota_atual: str | None,
    volume: float,
    mao_detectada: bool,
) -> None:
    """Renderiza painel superior informativo e barra vertical VU Meter para apresentação."""
    altura, largura = frame.shape[:2]

    # Painel superior translúcido
    altura_painel = 86
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (largura, altura_painel), (18, 20, 28), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.line(frame, (0, altura_painel), (largura, altura_painel), (70, 80, 100), 1)

    nome_nota = nota_atual if nota_atual is not None else "SILENCIO"
    freq = audio_engine.NOTAS_FREQUENCIAS.get(nota_atual) if nota_atual else None
    texto_freq = f"{freq:.2f} Hz" if freq is not None else "-- Hz"
    percentual = int(round(volume * 100))

    cor_nota = (80, 235, 140) if nota_atual is not None else (160, 165, 180)
    cor_status = (60, 220, 90) if mao_detectada else (0, 175, 255)

    # Indicador de detecção da mão
    cv2.circle(frame, (26, 43), 8, cor_status, -1)
    cv2.putText(
        frame,
        "NOTA",
        (50, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (175, 185, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        nome_nota,
        (50, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        cor_nota,
        2,
        cv2.LINE_AA,
    )

    # Frequência (Hz)
    x_freq = int(largura * 0.36)
    cv2.putText(
        frame,
        "FREQUENCIA",
        (x_freq, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (175, 185, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        texto_freq,
        (x_freq, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (240, 244, 250),
        2,
        cv2.LINE_AA,
    )

    # Intensidade (%)
    x_vol = int(largura * 0.68)
    cor_vol = _cor_vu_meter(volume)
    cv2.putText(
        frame,
        "INTENSIDADE",
        (x_vol, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (175, 185, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{percentual:3d}%",
        (x_vol, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        cor_vol,
        2,
        cv2.LINE_AA,
    )

    # Barra vertical VU Meter na lateral direita
    vu_largura = 32
    vu_x2 = largura - 24
    vu_x1 = vu_x2 - vu_largura
    vu_y1 = altura_painel + 36
    vu_y2 = altura - 54

    overlay_vu = frame.copy()
    cv2.rectangle(overlay_vu, (vu_x1 - 12, vu_y1 - 24), (vu_x2 + 10, vu_y2 + 28), (18, 20, 28), -1)
    cv2.addWeighted(overlay_vu, 0.65, frame, 0.35, 0, frame)

    cv2.putText(
        frame,
        "VOL",
        (vu_x1 - 2, vu_y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 218, 230),
        1,
        cv2.LINE_AA,
    )

    # Fundo interno da barra
    cv2.rectangle(frame, (vu_x1, vu_y1), (vu_x2, vu_y2), (45, 50, 62), -1)

    # Preenchimento proporcional de baixo para cima em segmentos coloridos
    altura_barra = vu_y2 - vu_y1
    progresso = max(0.0, min(1.0, (volume - VOL_MIN) / (VOL_MAX - VOL_MIN)))
    altura_preenchida = int(round(altura_barra * (0.10 + 0.90 * progresso)))
    topo_preenchido = vu_y2 - altura_preenchida

    num_segmentos = 24
    espaco = 2
    alt_seg = max(2, (altura_barra - (num_segmentos - 1) * espaco) // num_segmentos)
    for idx in range(num_segmentos):
        seg_y2 = vu_y2 - idx * (alt_seg + espaco)
        seg_y1 = max(vu_y1, seg_y2 - alt_seg)
        if seg_y2 <= topo_preenchido:
            continue
        nivel_seg = VOL_MIN + (idx / max(1, num_segmentos - 1)) * (VOL_MAX - VOL_MIN)
        cv2.rectangle(frame, (vu_x1 + 3, seg_y1), (vu_x2 - 3, seg_y2), _cor_vu_meter(nivel_seg), -1)

    cv2.rectangle(frame, (vu_x1, vu_y1), (vu_x2, vu_y2), (130, 140, 160), 1)
    cv2.putText(
        frame,
        f"{percentual}%",
        (vu_x1 - 4, vu_y2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        cor_vol,
        1,
        cv2.LINE_AA,
    )

    # Rodapé discreto
    cv2.putText(
        frame,
        "ESC ou Q: Sair  |  Altura do punho: Volume  |  Punho fechado: Silencio",
        (16, altura - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210, 215, 225),
        1,
        cv2.LINE_AA,
    )


def executar() -> None:
    """Executa o loop principal de captura de vídeo, detecção e síntese sonora."""
    audio_engine.inicializar_audio()

    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        raise RuntimeError(
            "Não foi possível acessar a webcam (índice 0). "
            "Verifique se a câmera está conectada e livre para uso."
        )

    mp_hands = mp.solutions.hands
    mp_desenho = mp.solutions.drawing_utils
    mp_estilos = mp.solutions.drawing_styles

    vol_suave = 0.55
    nota_atual: str | None = None
    candidato_estado: str | None = None
    contador_debounce: int = 0

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
        ) as hands:
            while True:
                sucesso, frame = captura.read()
                if not sucesso:
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                resultado = hands.process(frame_rgb)
                frame_rgb.flags.writeable = True

                mao_detectada = bool(resultado.multi_hand_landmarks)
                estado_valido = False
                proximo_estado: str | None = None

                if mao_detectada:
                    hand_landmarks = resultado.multi_hand_landmarks[0]
                    mp_desenho.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_estilos.get_default_hand_landmarks_style(),
                        mp_estilos.get_default_hand_connections_style(),
                    )

                    landmarks = hand_landmarks.landmark
                    vol_suave = calcular_volume_por_altura(landmarks[PUNHO].y, vol_suave)
                    audio_engine.atualizar_volume(vol_suave)

                    estado_valido, proximo_estado = classificar_gesto(landmarks)
                else:
                    # Sem mão no quadro: confirma transição para silêncio após debounce
                    estado_valido = True
                    proximo_estado = None

                if estado_valido:
                    if proximo_estado == candidato_estado:
                        contador_debounce += 1
                    else:
                        candidato_estado = proximo_estado
                        contador_debounce = 1

                    if contador_debounce >= FRAMES_CONFIRMACAO and candidato_estado != nota_atual:
                        nota_atual = candidato_estado
                        if nota_atual is None:
                            audio_engine.parar_nota()
                        else:
                            audio_engine.tocar_nota(nota_atual, vol_suave)

                desenhar_hud(frame, nota_atual, vol_suave, mao_detectada)
                cv2.imshow("Theremin Gestual - Dó Maior", frame)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (27, ord("q"), ord("Q")):
                    break
    finally:
        captura.release()
        cv2.destroyAllWindows()
        audio_engine.encerrar_audio()


if __name__ == "__main__":
    executar()
