"""Theremin Gestual — Instrumento musical via webcam com memes, timbres e modo 2 mãos.

Reconhece gestos geométricos da mão via MediaPipe Hands para tocar notas da
escala de Dó Maior, modula a intensidade sonora pela altura do punho, avalia
sequências musicais em tempo real exibindo memes animados e suporta timbres
selecionáveis e controle opcional com duas mãos.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import cv2
import mediapipe as mp
import numpy as np

import audio_engine
from meme_overlay import GerenciadorMemes
from music_evaluator import AvaliadorMusical

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


def _distancia_2d(lm_a: Any, lm_b: Any) -> float:
    """Calcula a distância euclidiana normalizada 2D entre dois landmarks."""
    return math.hypot(lm_a.x - lm_b.x, lm_a.y - lm_b.y)


def _polegar_aberto(landmarks: Sequence[Any]) -> bool:
    """Verifica se o polegar está estendido para fora da palma (ambidestro)."""
    dist_ponta_mindinho = _distancia_2d(landmarks[POLEGAR_TIP], landmarks[MINDINHO_MCP])
    dist_articulacao_mindinho = _distancia_2d(landmarks[POLEGAR_IP], landmarks[MINDINHO_MCP])
    dist_base_mindinho = _distancia_2d(landmarks[POLEGAR_MCP], landmarks[MINDINHO_MCP])
    return (
        dist_ponta_mindinho > dist_articulacao_mindinho
        and dist_ponta_mindinho > dist_base_mindinho
    )


def classificar_gesto(landmarks: Sequence[Any]) -> tuple[bool, str | None]:
    """Classifica os 21 landmarks em uma nota musical ou silêncio."""
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


def contar_dedos_abertos(landmarks: Sequence[Any]) -> int:
    """Conta quantos dedos (0 a 5) estão estendidos na mão."""
    abertos = sum(1 for tip, pip in PARES_DEDOS if landmarks[tip].y < landmarks[pip].y)
    if _polegar_aberto(landmarks):
        abertos += 1
    return abertos


def classificar_oitava(landmarks: Sequence[Any]) -> int:
    """Mapeia quantidade de dedos da mão esquerda para oitava (-1, 0, +1)."""
    dedos = contar_dedos_abertos(landmarks)
    if dedos <= 1:
        return -1
    if dedos <= 3:
        return 0
    return 1


def separar_maos_espelhadas(
    maos_landmarks: Sequence[Any],
    handedness_list: Sequence[Any] | None,
) -> tuple[Any | None, Any | None]:
    """Retorna ``(mao_direita_real, mao_esquerda_real)`` para imagem espelhada."""
    if not maos_landmarks:
        return None, None

    if len(maos_landmarks) == 1:
        rotulo = (
            handedness_list[0].classification[0].label
            if handedness_list and handedness_list[0].classification
            else "Left"
        )
        # Em imagem espelhada (cv2.flip), "Left" do MediaPipe é a mão direita real
        if rotulo == "Left":
            return maos_landmarks[0], None
        return None, maos_landmarks[0]

    candidatos = []
    for idx, hand_lm in enumerate(maos_landmarks[:2]):
        rotulo = (
            handedness_list[idx].classification[0].label
            if handedness_list and idx < len(handedness_list) and handedness_list[idx].classification
            else ""
        )
        candidatos.append((rotulo, hand_lm))

    mao_dir = next((lm for rot, lm in candidatos if rot == "Left"), None)
    mao_esq = next((lm for rot, lm in candidatos if rot == "Right"), None)

    if mao_dir is None or mao_esq is None:
        ordenado = sorted(maos_landmarks[:2], key=lambda h: h.landmark[PUNHO].x)
        mao_esq, mao_dir = ordenado[0], ordenado[1]

    return mao_dir, mao_esq


def _cor_vu_meter(nivel: float) -> tuple[int, int, int]:
    """Interpola cor BGR de verde -> amarelo -> vermelho conforme o nível [0.1, 1.0]."""
    progresso = max(0.0, min(1.0, (nivel - VOL_MIN) / (VOL_MAX - VOL_MIN)))
    if progresso < 0.5:
        t = progresso * 2.0
        b = int(60 * (1.0 - t))
        g = int(220 + 5 * t)
        r = int(60 + 195 * t)
    else:
        t = (progresso - 0.5) * 2.0
        b = int(50 * t)
        g = int(225 * (1.0 - t) + 60 * t)
        r = int(255 - 10 * t)
    return (b, g, r)


def desenhar_hud(
    frame: np.ndarray,
    nota_atual: str | None,
    volume: float,
    mao_detectada: bool,
    timbre: str = "theremin",
    oitava: int = 0,
    modo_duas_maos: bool = False,
    trilha_notas: Sequence[str] = (),
) -> None:
    """Renderiza painel superior, trilha de notas e VU Meter vertical."""
    altura, largura = frame.shape[:2]

    altura_painel = 86
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (largura, altura_painel), (18, 20, 28), -1)
    cv2.addWeighted(overlay, 0.74, frame, 0.26, 0, frame)
    cv2.line(frame, (0, altura_painel), (largura, altura_painel), (70, 80, 100), 1)

    nome_nota = nota_atual if nota_atual is not None else "SILENCIO"
    freq = (
        audio_engine.NOTAS_FREQUENCIAS[nota_atual] * (2.0 ** oitava)
        if nota_atual in audio_engine.NOTAS_FREQUENCIAS
        else None
    )
    texto_freq = f"{freq:.2f} Hz" if freq is not None else "-- Hz"
    percentual = int(round(volume * 100))

    cor_nota = (80, 235, 140) if nota_atual is not None else (160, 165, 180)
    cor_status = (60, 220, 90) if mao_detectada else (0, 175, 255)

    cv2.circle(frame, (22, 43), 8, cor_status, -1)
    cv2.putText(frame, "NOTA", (42, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (175, 185, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, nome_nota, (42, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.05, cor_nota, 2, cv2.LINE_AA)

    x_freq = int(largura * 0.28)
    cv2.putText(frame, "FREQUENCIA", (x_freq, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (175, 185, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, texto_freq, (x_freq, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 244, 250), 2, cv2.LINE_AA)

    x_cfg = int(largura * 0.54)
    nome_timbre = audio_engine.NOMES_TIMBRES.get(timbre, timbre)
    texto_oitava = f"{oitava:+d}" if oitava != 0 else "0"
    texto_modo = "2 MAOS" if modo_duas_maos else "1 MAO"
    cv2.putText(frame, f"TIMBRE | {texto_modo}", (x_cfg, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (175, 185, 200), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"{nome_timbre} (8va {texto_oitava})",
        (x_cfg, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 215, 110),
        2,
        cv2.LINE_AA,
    )

    x_vol = int(largura * 0.82)
    cor_vol = _cor_vu_meter(volume)
    cv2.putText(frame, "VOLUME", (x_vol, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (175, 185, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{percentual:3d}%", (x_vol, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.92, cor_vol, 2, cv2.LINE_AA)

    # Barra vertical VU Meter na lateral direita
    vu_largura = 30
    vu_x2 = largura - 20
    vu_x1 = vu_x2 - vu_largura
    vu_y1 = altura_painel + 34
    vu_y2 = altura - 66

    overlay_vu = frame.copy()
    cv2.rectangle(overlay_vu, (vu_x1 - 12, vu_y1 - 24), (vu_x2 + 10, vu_y2 + 26), (18, 20, 28), -1)
    cv2.addWeighted(overlay_vu, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "VOL", (vu_x1 - 2, vu_y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 218, 230), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (vu_x1, vu_y1), (vu_x2, vu_y2), (45, 50, 62), -1)

    altura_barra = vu_y2 - vu_y1
    progresso = max(0.0, min(1.0, (volume - VOL_MIN) / (VOL_MAX - VOL_MIN)))
    altura_preenchida = int(round(altura_barra * (0.10 + 0.90 * progresso)))
    topo_preenchido = vu_y2 - altura_preenchida

    num_segmentos = 22
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
    cv2.putText(frame, f"{percentual}%", (vu_x1 - 4, vu_y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, cor_vol, 1, cv2.LINE_AA)

    # Trilha visual das últimas notas tocadas
    if trilha_notas:
        trilha_str = "Trilha: " + " -> ".join(trilha_notas)
        cv2.putText(frame, trilha_str, (16, altura - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (125, 225, 255), 1, cv2.LINE_AA)

    cv2.putText(
        frame,
        "1-3: Timbre | M: Modo 2 Maos | C: Limpar Trilha | ESC/Q: Sair",
        (16, altura - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (210, 215, 225),
        1,
        cv2.LINE_AA,
    )


def executar() -> None:
    """Executa o loop principal de captura de vídeo, detecção e síntese sonora."""
    audio_engine.inicializar_audio()
    avaliador = AvaliadorMusical()
    memes = GerenciadorMemes()

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
    modo_duas_maos = False

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
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

                maos = resultado.multi_hand_landmarks or []
                mao_detectada = len(maos) > 0
                for hand_lm in maos:
                    mp_desenho.draw_landmarks(
                        frame,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_estilos.get_default_hand_landmarks_style(),
                        mp_estilos.get_default_hand_connections_style(),
                    )

                estado_valido = False
                proximo_estado: str | None = None

                if mao_detectada:
                    if modo_duas_maos:
                        mao_dir, mao_esq = separar_maos_espelhadas(maos, resultado.multi_handedness)
                        if mao_esq is not None:
                            vol_suave = calcular_volume_por_altura(mao_esq.landmark[PUNHO].y, vol_suave)
                            audio_engine.atualizar_volume(vol_suave)
                            audio_engine.definir_oitava(classificar_oitava(mao_esq.landmark))
                        if mao_dir is not None:
                            estado_valido, proximo_estado = classificar_gesto(mao_dir.landmark)
                        else:
                            estado_valido, proximo_estado = True, None
                    else:
                        landmarks = maos[0].landmark
                        vol_suave = calcular_volume_por_altura(landmarks[PUNHO].y, vol_suave)
                        audio_engine.atualizar_volume(vol_suave)
                        estado_valido, proximo_estado = classificar_gesto(landmarks)
                else:
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
                            avaliacao = avaliador.registrar_nota(nota_atual)
                            if avaliacao is not None and avaliacao.categoria is not None:
                                memes.ativar(avaliacao.categoria, avaliacao.motivo)
                                audio_engine.tocar_sfx_meme(avaliacao.categoria)

                timbre_atual, oitava_atual = audio_engine.obter_estado_audio()
                desenhar_hud(
                    frame,
                    nota_atual,
                    vol_suave,
                    mao_detectada,
                    timbre=timbre_atual,
                    oitava=oitava_atual,
                    modo_duas_maos=modo_duas_maos,
                    trilha_notas=avaliador.historico,
                )
                memes.renderizar(frame)
                cv2.imshow("Theremin Gestual - Dó Maior", frame)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (27, ord("q"), ord("Q")):
                    break
                if tecla in (ord("1"), ord("2"), ord("3")):
                    audio_engine.definir_timbre(tecla - ord("0"))
                elif tecla in (ord("m"), ord("M")):
                    modo_duas_maos = not modo_duas_maos
                    if not modo_duas_maos:
                        audio_engine.definir_oitava(0)
                elif tecla in (ord("c"), ord("C")):
                    avaliador.limpar()
    finally:
        captura.release()
        cv2.destroyAllWindows()
        audio_engine.encerrar_audio()


if __name__ == "__main__":
    executar()
