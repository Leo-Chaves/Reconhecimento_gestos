"""Tocador Musical Gestual — Instrumento interativo HD com 5 timbres, arpejador e looper.

Adapta dinamicamente os controles conforme o número de mãos detectadas:
- 1 Mão (Simplificado): mapeia as 8 notas (DO4 a DO5) no Modo Maior padrão.
- 2 Mãos (Expressivo):
  * Mão Esquerda: seleciona a nota musical (DO4 a DO5) ou Silêncio.
  * Mão Direita: controla o volume (eixo Y) e alterna MAIOR (palma aberta) vs.
    MENOR (punho fechado ou pinça).
  * Aproximar as mãos na horizontal: ativa o Arpejador com corda luminosa.

Suporta janela expansível / Tela Cheia (F), touch aéreo na barra superior,
Mini-Looper (R), guia sob demanda (G ou botão [?]) e pop-ups de gatos.
"""

from __future__ import annotations

import time
from typing import Sequence

import cv2
import mediapipe as mp

import audio_engine
from hand_tracker import RastreadorMaos, TipoEscala
from looper import MiniLooper
from meme_overlay import SistemaParticulasMemes
import ui_renderer
from ui_renderer import BotaoUI

# ---------------------------------------------------------------------------
# Parâmetros de Calibração: Debounce Temporal e Touch Aéreo
# ---------------------------------------------------------------------------
FRAMES_CONFIRMACAO_NOTA: int = 3
TEMPO_ESTABILIDADE_SEGUNDOS: float = 0.055
TEMPO_DWELL_TOUCH: float = 0.40
COOLDOWN_TOUCH: float = 0.70

LARGURA_CANVAS: int = 1280
ALTURA_CANVAS: int = 720
TITULO_JANELA: str = "Tocador Musical Gestual"


class EstabilizadorNota:
    """Filtro temporal combinado (frames + tempo) para confirmação de notas."""

    def __init__(
        self,
        min_frames: int = FRAMES_CONFIRMACAO_NOTA,
        min_segundos: float = TEMPO_ESTABILIDADE_SEGUNDOS,
    ) -> None:
        self.min_frames = max(1, int(min_frames))
        self.min_segundos = max(0.0, float(min_segundos))
        self._candidato: str | None = None
        self._contador: int = 0
        self._inicio_candidato: float = 0.0
        self.nota_confirmada: str | None = None

    def atualizar(
        self,
        gesto_valido: bool,
        nota_candidata: str | None,
        agora: float,
    ) -> str | None:
        """Atualiza o buffer e retorna a nota atualmente confirmada."""
        if not gesto_valido:
            self._contador = 0
            return self.nota_confirmada

        if self._contador > 0 and nota_candidata == self._candidato:
            self._contador += 1
        else:
            self._candidato = nota_candidata
            self._contador = 1
            self._inicio_candidato = agora

        if (
            self._contador >= self.min_frames
            and (agora - self._inicio_candidato) >= self.min_segundos
        ):
            self.nota_confirmada = self._candidato

        return self.nota_confirmada


class GerenciadorTouchAereo:
    """Detecta permanência (dwell) da ponta do indicador sobre botões da HUD."""

    def __init__(
        self,
        tempo_dwell: float = TEMPO_DWELL_TOUCH,
        cooldown: float = COOLDOWN_TOUCH,
    ) -> None:
        self.tempo_dwell = max(0.05, float(tempo_dwell))
        self.cooldown = max(0.0, float(cooldown))
        self._hover_id: str | None = None
        self._hover_inicio: float = 0.0
        self._ultimo_disparo: float = -999.0

    def atualizar(
        self,
        pontas_norm: Sequence[tuple[float, float]],
        botoes: Sequence[BotaoUI],
        largura: int,
        altura: int,
        agora: float,
    ) -> tuple[str | None, str | None, float]:
        """Retorna ``(acao_disparada, id_hover_atual, progresso_0_a_1)``."""
        if not pontas_norm or not botoes or (agora - self._ultimo_disparo) < self.cooldown:
            self._hover_id = None
            return None, None, 0.0

        alvo_id: str | None = None
        for nx, ny in pontas_norm:
            px = int(round(nx * largura))
            py = int(round(ny * altura))
            for btn in botoes:
                if btn.contem(px, py):
                    alvo_id = btn.id_acao
                    break
            if alvo_id is not None:
                break

        if alvo_id is None:
            self._hover_id = None
            return None, None, 0.0

        if alvo_id != self._hover_id:
            self._hover_id = alvo_id
            self._hover_inicio = agora
            return None, alvo_id, 0.0

        decorrido = agora - self._hover_inicio
        if decorrido >= self.tempo_dwell:
            self._ultimo_disparo = agora
            self._hover_id = None
            return alvo_id, None, 0.0

        return None, alvo_id, max(0.0, min(1.0, decorrido / self.tempo_dwell))


def executar() -> None:
    """Executa o loop principal HD não-bloqueante com áudio, arpejador e HUD."""
    audio_engine.inicializar_audio()
    rastreador = RastreadorMaos()
    estabilizador = EstabilizadorNota()
    touch_aereo = GerenciadorTouchAereo()
    looper = MiniLooper()
    memes = SistemaParticulasMemes()
    faiscas_corda = ui_renderer.SistemaFaiscasCorda()

    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        audio_engine.encerrar_audio()
        raise RuntimeError(
            "Não foi possível acessar a webcam (índice 0). "
            "Verifique se a câmera está conectada e livre para uso."
        )

    captura.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA_CANVAS)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA_CANVAS)

    cv2.namedWindow(TITULO_JANELA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TITULO_JANELA, LARGURA_CANVAS, ALTURA_CANVAS)

    clique_mouse: list[tuple[int, int] | None] = [None]

    def _on_mouse(evento: int, x: int, y: int, _flags: int, _param: object) -> None:
        if evento == cv2.EVENT_LBUTTONDOWN:
            clique_mouse[0] = (x, y)

    cv2.setMouseCallback(TITULO_JANELA, _on_mouse)

    mp_hands = mp.solutions.hands

    nota_tocando: str | None = None
    escala_tocando: TipoEscala = "maior"
    arpejo_tocando: bool = False
    passo_arpejo: int = 0
    grau_ativo_arpejo: int = 0
    ultimo_pulso_arpejo: float = 0.0
    proximo_pulso_arpejo: float = 0.0

    mostrar_guia: bool = False
    tela_cheia: bool = False
    botoes_ui: list[BotaoUI] = []

    def _executar_acao_ui(acao: str, agora_ev: float, vol_ev: float, arp_ev: bool) -> None:
        nonlocal mostrar_guia, passo_arpejo
        if acao.startswith("timbre:"):
            indice = int(acao.split(":")[1])
            audio_engine.definir_timbre(indice)
        elif acao == "arpejo:padrao":
            audio_engine.alternar_padrao_arpejo()
            passo_arpejo = 0
        elif acao == "looper:toggle":
            tim_atual, _ = audio_engine.obter_estado_audio()
            looper.alternar(
                agora_ev,
                nota_atual=nota_tocando,
                escala_atual=escala_tocando,
                timbre_atual=tim_atual,
                volume_atual=vol_ev,
                arpejo_atual=arp_ev,
            )
        elif acao == "ajuda:toggle":
            mostrar_guia = not mostrar_guia

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.72,
            min_tracking_confidence=0.72,
        ) as hands:
            while True:
                sucesso, frame = captura.read()
                if not sucesso:
                    break

                agora = time.perf_counter()

                frame = cv2.flip(frame, 1)
                if frame.shape[1] < LARGURA_CANVAS or frame.shape[0] < ALTURA_CANVAS:
                    frame = cv2.resize(frame, (LARGURA_CANVAS, ALTURA_CANVAS), interpolation=cv2.INTER_LINEAR)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                resultado = hands.process(frame_rgb)
                frame_rgb.flags.writeable = True

                maos_detectadas = resultado.multi_hand_landmarks or []
                estado_maos = rastreador.processar(
                    maos_detectadas,
                    resultado.multi_handedness,
                    agora=agora,
                )

                # Processa cliques do mouse ou Touch Aéreo nos botões da barra superior
                if clique_mouse[0] is not None:
                    mx, my = clique_mouse[0]
                    clique_mouse[0] = None
                    for btn in botoes_ui:
                        if btn.contem(mx, my):
                            _executar_acao_ui(btn.id_acao, agora, estado_maos.volume, estado_maos.arpejo_ativo)
                            break

                acao_touch, hover_id, prog_hover = touch_aereo.atualizar(
                    estado_maos.pontas_indicadores,
                    botoes_ui,
                    frame.shape[1],
                    frame.shape[0],
                    agora,
                )
                if acao_touch is not None:
                    _executar_acao_ui(acao_touch, agora, estado_maos.volume, estado_maos.arpejo_ativo)

                nota_confirmada = estabilizador.atualizar(
                    estado_maos.gesto_valido,
                    estado_maos.nota_candidata,
                    agora,
                )

                timbre_atual, modo_harmonico_atual = audio_engine.obter_estado_audio()
                padrao_arpejo_atual = audio_engine.obter_padrao_arpejo()
                disparou_pulso_frame = False

                if nota_confirmada is not None:
                    mudou_nota = nota_confirmada != nota_tocando
                    mudou_escala = estado_maos.escala != escala_tocando
                    mudou_modo_arp = estado_maos.arpejo_ativo != arpejo_tocando

                    if estado_maos.arpejo_ativo:
                        if mudou_nota or mudou_escala or mudou_modo_arp:
                            audio_engine.tocar_nota(
                                nota_confirmada,
                                escala=estado_maos.escala,
                                volume=estado_maos.volume,
                                arpejo=True,
                            )
                            passo_arpejo = 0
                            proximo_pulso_arpejo = agora
                            if mudou_nota or mudou_escala:
                                memes.disparar(
                                    estado_maos.escala,
                                    posicao_norm=estado_maos.ancora_meme,
                                    rotulo=f"{nota_confirmada} {estado_maos.escala.upper()}",
                                    agora=agora,
                                )
                            nota_tocando = nota_confirmada
                            escala_tocando = estado_maos.escala
                            arpejo_tocando = True

                        if agora >= proximo_pulso_arpejo:
                            passo_usado = passo_arpejo
                            passo_arpejo, grau_ativo_arpejo = audio_engine.disparar_passo_arpejo(
                                nota_confirmada,
                                escala=estado_maos.escala,
                                passo=passo_usado,
                                volume=estado_maos.volume,
                            )
                            looper.registrar_passo_arpejo(
                                nota_confirmada,
                                estado_maos.escala,
                                timbre_atual,
                                passo_usado,
                                estado_maos.volume,
                                agora,
                                padrao_arpejo=padrao_arpejo_atual,
                            )
                            ultimo_pulso_arpejo = agora
                            proximo_pulso_arpejo = agora + (1.0 / max(1.0, estado_maos.taxa_arpejo_hz))
                            disparou_pulso_frame = True
                    else:
                        if mudou_nota or mudou_escala or mudou_modo_arp:
                            audio_engine.tocar_nota(
                                nota_confirmada,
                                escala=estado_maos.escala,
                                volume=estado_maos.volume,
                                arpejo=False,
                            )
                            looper.registrar_nota(
                                nota_confirmada,
                                estado_maos.escala,
                                timbre_atual,
                                estado_maos.volume,
                                agora,
                            )
                            if mudou_nota or mudou_escala:
                                memes.disparar(
                                    estado_maos.escala,
                                    posicao_norm=estado_maos.ancora_meme,
                                    rotulo=f"{nota_confirmada} {estado_maos.escala.upper()}",
                                    agora=agora,
                                )
                            nota_tocando = nota_confirmada
                            escala_tocando = estado_maos.escala
                            arpejo_tocando = False
                        else:
                            audio_engine.atualizar_volume(estado_maos.volume)
                else:
                    if nota_tocando is not None:
                        audio_engine.parar_nota()
                        looper.registrar_parada(agora)
                        nota_tocando = None
                        arpejo_tocando = False

                looper.atualizar(agora)

                ui_renderer.desenhar_maos(frame, estado_maos.maos_visuais)
                if estado_maos.modo_duas_maos and estado_maos.arpejo_ativo:
                    nos_corda = ui_renderer.desenhar_corda_arpejo(
                        frame,
                        estado_maos.ponto_palma_esq,
                        estado_maos.ponto_palma_dir,
                        estado_maos.escala,
                        nota_tocando,
                        agora - ultimo_pulso_arpejo,
                        agora,
                        grau_ativo=grau_ativo_arpejo,
                    )
                    if disparou_pulso_frame and nos_corda:
                        nx_px, ny_px = nos_corda[grau_ativo_arpejo % len(nos_corda)]
                        cor_faisca = (145, 250, 190) if estado_maos.escala == "maior" else (255, 215, 125)
                        faiscas_corda.emitir(nx_px, ny_px, cor_faisca, quantidade=7, agora=agora)

                faiscas_corda.atualizar_e_renderizar(frame, agora=agora)
                memes.atualizar_e_renderizar(frame, agora=agora)

                botoes_ui = ui_renderer.desenhar_hud(
                    frame,
                    estado_maos=estado_maos,
                    nota_ativa=nota_tocando,
                    timbre=timbre_atual,
                    modo_harmonico=modo_harmonico_atual,
                    looper_estado=looper.estado,
                    looper_progresso=looper.obter_progresso(agora),
                    mostrar_guia=mostrar_guia,
                    botao_hover_id=hover_id,
                    progresso_hover=prog_hover,
                    padrao_arpejo=padrao_arpejo_atual,
                )

                cv2.imshow(TITULO_JANELA, frame)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (27, ord("q"), ord("Q")):
                    break
                if tecla in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                    audio_engine.definir_timbre(tecla - ord("0"))
                elif tecla in (ord("a"), ord("A")):
                    _executar_acao_ui("arpejo:padrao", agora, estado_maos.volume, estado_maos.arpejo_ativo)
                elif tecla in (ord("h"), ord("H")):
                    audio_engine.alternar_modo_harmonico()
                elif tecla in (ord("g"), ord("G")):
                    mostrar_guia = not mostrar_guia
                elif tecla in (ord("r"), ord("R")):
                    _executar_acao_ui("looper:toggle", agora, estado_maos.volume, estado_maos.arpejo_ativo)
                elif tecla in (ord("f"), ord("F")):
                    tela_cheia = not tela_cheia
                    cv2.setWindowProperty(
                        TITULO_JANELA,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if tela_cheia else cv2.WINDOW_NORMAL,
                    )
    finally:
        looper.parar()
        captura.release()
        cv2.destroyAllWindows()
        audio_engine.encerrar_audio()


if __name__ == "__main__":
    executar()
