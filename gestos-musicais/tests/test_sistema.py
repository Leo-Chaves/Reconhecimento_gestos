"""Testes automatizados para todos os módulos do Tocador Musical Gestual."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import time
import unittest

import cv2
import numpy as np

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

import audio_engine  # noqa: E402
from hand_tracker import (  # noqa: E402
    RastreadorMaos,
    VOLUME_PADRAO_1_MAO,
    classificar_escala_mao_direita,
    classificar_gesto_nota,
    detectar_dedos_com_histerese,
    separar_duas_maos,
)
from looper import MiniLooper  # noqa: E402
from main import EstabilizadorNota, GerenciadorTouchAereo  # noqa: E402
from meme_overlay import SistemaParticulasMemes, calcular_curva_animacao  # noqa: E402
import ui_renderer  # noqa: E402


@dataclass
class _FakeClassificacao:
    label: str


@dataclass
class _FakeHandedness:
    classification: list[_FakeClassificacao]


def _rotacionar_mao(pts: np.ndarray, angulo_graus: float) -> np.ndarray:
    """Rotaciona os 21 landmarks ao redor do punho (índice 0)."""
    if abs(angulo_graus) < 1e-6:
        return pts
    rad = math.radians(angulo_graus)
    c, s = math.cos(rad), math.sin(rad)
    pivo = pts[0].copy()
    desloc = pts - pivo
    rot = np.empty_like(desloc)
    rot[:, 0] = desloc[:, 0] * c - desloc[:, 1] * s
    rot[:, 1] = desloc[:, 0] * s + desloc[:, 1] * c
    return rot + pivo


def _criar_mao_sintetica(
    dedos_abertos: tuple[bool, bool, bool, bool],
    polegar_aberto: bool = False,
    pinca: bool = False,
    palma_para_camera: bool = True,
    lado: str = "esquerda",
    centro_x: float = 0.28,
    punho_y: float = 0.72,
    angulo_graus: float = 0.0,
) -> np.ndarray:
    """Constrói um esqueleto realista de 21 landmarks normalizados."""
    pts = np.zeros((21, 2), dtype=np.float32)
    pts[0] = [centro_x, punho_y]

    if lado == "esquerda":
        sinal_ind = -1.0 if palma_para_camera else 1.0
    else:
        sinal_ind = 1.0 if palma_para_camera else -1.0

    pts[1] = [centro_x + sinal_ind * 0.035, punho_y - 0.045]
    pts[2] = [centro_x + sinal_ind * 0.065, punho_y - 0.095]

    mcps = {
        5: (sinal_ind * 0.055, -0.155),
        9: (sinal_ind * 0.015, -0.170),
        13: (-sinal_ind * 0.025, -0.162),
        17: (-sinal_ind * 0.062, -0.145),
    }
    for idx, (dx, dy) in mcps.items():
        pts[idx] = [centro_x + dx, punho_y + dy]

    bases = (5, 9, 13, 17)
    for base, aberto in zip(bases, dedos_abertos):
        mx, my = pts[base]
        if aberto:
            pts[base + 1] = [mx, my - 0.055]
            pts[base + 2] = [mx, my - 0.098]
            pts[base + 3] = [mx, my - 0.138]
        else:
            pts[base + 1] = [mx, my - 0.035]
            pts[base + 2] = [mx, my + 0.020]
            pts[base + 3] = [mx, my + 0.055]

    if pinca:
        pts[6] = [pts[5, 0], pts[5, 1] - 0.055]
        pts[7] = [pts[5, 0] + sinal_ind * 0.015, pts[5, 1] - 0.090]
        pts[8] = [pts[5, 0] + sinal_ind * 0.025, pts[5, 1] - 0.105]
        pts[3] = [pts[2, 0] + sinal_ind * 0.015, pts[2, 1] - 0.060]
        pts[4] = [pts[8, 0] - sinal_ind * 0.008, pts[8, 1] + 0.006]
    elif polegar_aberto:
        pts[3] = [pts[2, 0] + sinal_ind * 0.045, pts[2, 1] - 0.035]
        pts[4] = [pts[2, 0] + sinal_ind * 0.088, pts[2, 1] - 0.062]
    else:
        pts[3] = [pts[2, 0] - sinal_ind * 0.020, pts[2, 1] - 0.015]
        pts[4] = [pts[2, 0] - sinal_ind * 0.045, pts[2, 1] + 0.010]

    return _rotacionar_mao(pts, angulo_graus)


class TestHandTracker(unittest.TestCase):
    """Testes das 8 notas, histerese, postura Maior/Menor e Arpejador espacial."""

    def test_classificacao_oito_notas_e_inclinacao(self) -> None:
        casos = [
            ((True, False, False, False), False, "DO4"),
            ((True, True, False, False), False, "RE4"),
            ((True, True, True, False), False, "MI4"),
            ((True, True, True, True), False, "FA4"),
            ((True, True, True, True), True, "SOL4"),
            ((False, False, False, True), True, "LA4"),
            ((True, False, False, True), False, "SI4"),
            ((True, False, False, True), True, "SI4"),
            ((True, False, False, False), True, "DO5"),
            ((False, False, False, False), False, None),
            ((False, False, False, False), True, None),
        ]
        for angulo in (0.0, 30.0, -30.0):
            for dedos, polegar, esperado in casos:
                mao = _criar_mao_sintetica(
                    dedos_abertos=dedos,
                    polegar_aberto=polegar,
                    angulo_graus=angulo,
                )
                valido, nota = classificar_gesto_nota(mao)
                self.assertTrue(valido)
                self.assertEqual(nota, esperado)

    def test_histerese_dedos_normalizada_pela_palma(self) -> None:
        mao_base = _criar_mao_sintetica((True, False, False, False), polegar_aberto=False)
        tam_palma = float(np.hypot(*(mao_base[9] - mao_base[0])))
        dist_pip = float(np.hypot(*(mao_base[6] - mao_base[0])))

        mao_zona_morta = mao_base.copy()
        mao_zona_morta[8] = [mao_base[0, 0], mao_base[0, 1] - (dist_pip + 0.11 * tam_palma)]

        dedos_vindo_aberto = detectar_dedos_com_histerese(
            mao_zona_morta,
            anterior=(False, True, False, False, False),
        )
        dedos_vindo_fechado = detectar_dedos_com_histerese(
            mao_zona_morta,
            anterior=(False, False, False, False, False),
        )
        self.assertTrue(dedos_vindo_aberto[1])
        self.assertFalse(dedos_vindo_fechado[1])

    def test_postura_mao_direita_maior_vs_menor(self) -> None:
        mao_aberta = _criar_mao_sintetica((True, True, True, True), polegar_aberto=True, lado="direita", centro_x=0.75)
        mao_punho = _criar_mao_sintetica((False, False, False, False), polegar_aberto=False, lado="direita", centro_x=0.75)
        mao_pinca = _criar_mao_sintetica((False, True, True, True), pinca=True, lado="direita", centro_x=0.75)

        self.assertEqual(classificar_escala_mao_direita(mao_aberta, "menor")[0], "maior")
        self.assertEqual(classificar_escala_mao_direita(mao_punho, "maior")[0], "menor")
        self.assertEqual(classificar_escala_mao_direita(mao_pinca, "maior")[0], "menor")

    def test_separacao_maos_espelhada_e_fallback_espacial(self) -> None:
        esq = _criar_mao_sintetica((True, False, False, False), lado="esquerda", centro_x=0.25)
        dir_ = _criar_mao_sintetica((True, True, True, True), lado="direita", centro_x=0.75)

        handed = [_FakeHandedness([_FakeClassificacao("Right")]), _FakeHandedness([_FakeClassificacao("Left")])]
        out_esq, out_dir = separar_duas_maos([dir_, esq], handed)
        self.assertAlmostEqual(float(out_esq[0, 0]), 0.25, places=4)
        self.assertAlmostEqual(float(out_dir[0, 0]), 0.75, places=4)

        handed_dup = [_FakeHandedness([_FakeClassificacao("Left")]), _FakeHandedness([_FakeClassificacao("Left")])]
        out_esq2, out_dir2 = separar_duas_maos([dir_, esq], handed_dup)
        self.assertAlmostEqual(float(out_esq2[0, 0]), 0.25, places=4)
        self.assertAlmostEqual(float(out_dir2[0, 0]), 0.75, places=4)

    def test_arpejador_por_proximidade_horizontal_com_histerese(self) -> None:
        rastreador = RastreadorMaos()
        handed = [_FakeHandedness([_FakeClassificacao("Left")]), _FakeHandedness([_FakeClassificacao("Right")])]

        # Mãos afastadas (dist ~ 0.50 > 0.35): acorde cheio (arpejo desligado)
        esq = _criar_mao_sintetica((True, False, False, False), lado="esquerda", centro_x=0.25)
        dir_afastada = _criar_mao_sintetica((True, True, True, True), polegar_aberto=True, lado="direita", centro_x=0.75)
        est_longe = rastreador.processar([esq, dir_afastada], handed)
        self.assertFalse(est_longe.arpejo_ativo)
        self.assertEqual(est_longe.taxa_arpejo_hz, 0.0)
        self.assertEqual(len(est_longe.pontas_indicadores), 2)

        # Mãos próximas (dist ~ 0.25 <= 0.35): liga Arpejador
        dir_perto = _criar_mao_sintetica((True, True, True, True), polegar_aberto=True, lado="direita", centro_x=0.50)
        est_perto = rastreador.processar([esq, dir_perto], handed)
        self.assertTrue(est_perto.arpejo_ativo)
        self.assertGreater(est_perto.taxa_arpejo_hz, 5.0)

        # Ainda mais perto (dist ~ 0.16): velocidade aumenta
        dir_muito_perto = _criar_mao_sintetica((True, True, True, True), polegar_aberto=True, lado="direita", centro_x=0.41)
        est_rapido = rastreador.processar([esq, dir_muito_perto], handed)
        self.assertGreater(est_rapido.taxa_arpejo_hz, est_perto.taxa_arpejo_hz)

        # Zona de histerese (dist ~ 0.37): mantém ligado
        dir_hist = _criar_mao_sintetica((True, True, True, True), polegar_aberto=True, lado="direita", centro_x=0.59)
        est_hist = rastreador.processar([esq, dir_hist], handed)
        self.assertTrue(est_hist.arpejo_ativo)

        # Afasta acima de 0.39 (dist ~ 0.46): desliga
        dir_fora = _criar_mao_sintetica((True, True, True, True), polegar_aberto=True, lado="direita", centro_x=0.71)
        est_fora = rastreador.processar([esq, dir_fora], handed)
        self.assertFalse(est_fora.arpejo_ativo)

    def test_adaptacao_1_mao_vs_2_maos(self) -> None:
        rastreador = RastreadorMaos()
        mao_unica = _criar_mao_sintetica((True, False, False, False), polegar_aberto=True, punho_y=0.20)
        estado_1 = rastreador.processar([mao_unica], [_FakeHandedness([_FakeClassificacao("Left")])])
        self.assertFalse(estado_1.modo_duas_maos)
        self.assertEqual(estado_1.nota_candidata, "DO5")
        self.assertEqual(estado_1.escala, "maior")
        self.assertAlmostEqual(estado_1.volume, VOLUME_PADRAO_1_MAO, places=4)
        self.assertFalse(estado_1.arpejo_ativo)

    def test_memoria_anti_piscada_dois_frames(self) -> None:
        rastreador = RastreadorMaos()
        mao = _criar_mao_sintetica((True, True, False, False), lado="esquerda", centro_x=0.30)
        handed = [_FakeHandedness([_FakeClassificacao("Left")])]

        est_ok = rastreador.processar([mao], handed)
        self.assertEqual(est_ok.nota_candidata, "RE4")

        # 1º e 2º frames sem mão devem preservar o último estado válido
        est_piscada_1 = rastreador.processar([], [])
        est_piscada_2 = rastreador.processar([], [])
        self.assertEqual(est_piscada_1.nota_candidata, "RE4")
        self.assertEqual(est_piscada_2.nota_candidata, "RE4")

        # 3º frame consecutivo sem mão entra em silêncio real
        est_silencio = rastreador.processar([], [])
        self.assertEqual(est_silencio.num_maos, 0)
        self.assertIsNone(est_silencio.nota_candidata)


class TestEstabilizadorETouchAereo(unittest.TestCase):
    """Testes do debounce de notas e do dwell do Touch Aéreo na HUD."""

    def test_debounce_frames_e_tempo_e_transicao_invalida(self) -> None:
        estab = EstabilizadorNota(min_frames=3, min_segundos=0.055)
        self.assertIsNone(estab.atualizar(True, "DO4", agora=1.00))
        self.assertIsNone(estab.atualizar(True, "DO4", agora=1.03))
        self.assertEqual(estab.atualizar(True, "DO4", agora=1.06), "DO4")

        # Estado transitório inválido mantém DO4 e reinicia contagem do próximo candidato
        self.assertEqual(estab.atualizar(False, None, agora=1.09), "DO4")

        self.assertEqual(estab.atualizar(True, "SOL4", agora=1.12), "DO4")
        self.assertEqual(estab.atualizar(True, "SOL4", agora=1.15), "DO4")
        self.assertEqual(estab.atualizar(True, "SOL4", agora=1.18), "SOL4")

    def test_touch_aereo_dwell_e_cooldown(self) -> None:
        touch = GerenciadorTouchAereo(tempo_dwell=0.40, cooldown=0.70)
        botoes = [ui_renderer.BotaoUI("timbre:2", "2 Violao", 480, 16, 590, 54)]

        # Ponto central do botão em 1280x720 -> (535/1280, 35/720)
        ponta = [(535 / 1280.0, 35 / 720.0)]

        acao, hov, prog = touch.atualizar(ponta, botoes, 1280, 720, agora=1.00)
        self.assertIsNone(acao)
        self.assertEqual(hov, "timbre:2")
        self.assertEqual(prog, 0.0)

        acao, hov, prog = touch.atualizar(ponta, botoes, 1280, 720, agora=1.20)
        self.assertIsNone(acao)
        self.assertEqual(hov, "timbre:2")
        self.assertAlmostEqual(prog, 0.5, places=2)

        # Completa 0.40s -> dispara
        acao, hov, prog = touch.atualizar(ponta, botoes, 1280, 720, agora=1.41)
        self.assertEqual(acao, "timbre:2")
        self.assertIsNone(hov)

        # Cooldown impede re-disparo imediato
        acao2, _, _ = touch.atualizar(ponta, botoes, 1280, 720, agora=1.60)
        self.assertIsNone(acao2)


class TestAudioEngine(unittest.TestCase):
    """Testes dos 5 instrumentos, reverb estéreo, arpejador e looper."""

    def tearDown(self) -> None:
        audio_engine.encerrar_audio()

    def test_continuidade_fase_estereo_cinco_instrumentos(self) -> None:
        for timbre in audio_engine.TIMBRES:
            for modo in ("acorde", "nota"):
                for escala in ("maior", "menor"):
                    for nota in ("DO4", "DO5"):
                        estereo = audio_engine.gerar_buffer_onda(nota, escala, timbre, modo)
                        self.assertEqual(estereo.ndim, 2)
                        self.assertEqual(estereo.shape[1], 2)
                        for ch in (0, 1):
                            canal = estereo[:, ch]
                            passo = float(canal[1] - canal[0])
                            salto = abs(float(canal[0] - canal[-1]) - passo)
                            self.assertLess(salto, 0.03, f"Clique em {timbre}/{modo}/{escala}/{nota}/ch={ch}")

    def test_cinco_timbres_arpejador_e_canal_looper(self) -> None:
        audio_engine.inicializar_audio()
        for idx, nome in enumerate(audio_engine.TIMBRES, start=1):
            self.assertEqual(audio_engine.definir_timbre(idx), nome)
            audio_engine.tocar_nota("DO4", escala="maior", volume=0.8, arpejo=False)
            audio_engine.tocar_nota("DO4", escala="maior", volume=0.8, arpejo=True)
            p1 = audio_engine.disparar_passo_arpejo("DO4", escala="maior", passo=0, volume=0.8)
            p2 = audio_engine.disparar_passo_arpejo("DO4", escala="maior", passo=p1, volume=0.8)
            self.assertEqual((p1, p2), (1, 2))

        # Passagem fluida (Legato Maior -> Menor na mesma nota e troca de nota com release natural)
        audio_engine.definir_timbre("piano")
        audio_engine.tocar_nota("DO4", escala="maior", volume=0.8, arpejo=False)
        audio_engine.tocar_nota("DO4", escala="menor", volume=0.8, arpejo=False)
        audio_engine.tocar_nota("FA4", escala="maior", volume=0.8, arpejo=False)
        audio_engine.atualizar_volume(0.65)

        # Canal do looper independente
        audio_engine.tocar_nota("SOL4", escala="menor", volume=0.5, timbre_override="violao", canal_looper=True)
        audio_engine.disparar_passo_arpejo("SOL4", escala="menor", passo=0, volume=0.5, timbre_override="piano", canal_looper=True)
        audio_engine.parar_nota(canal_looper=False)
        audio_engine.parar_nota(canal_looper=True)


class TestMiniLooper(unittest.TestCase):
    """Testes do ciclo de gravação e reprodução contínua do MiniLooper."""

    def tearDown(self) -> None:
        audio_engine.encerrar_audio()

    def test_ciclo_gravar_tocar_e_parar(self) -> None:
        audio_engine.inicializar_audio()
        looper = MiniLooper()
        self.assertEqual(looper.estado, "parado")

        # Inicia gravação com DO4 já soando
        estado = looper.alternar(10.0, nota_atual="DO4", escala_atual="maior", timbre_atual="violao", volume_atual=0.8)
        self.assertEqual(estado, "gravando")
        looper.registrar_nota("SOL4", "maior", "violao", 0.8, agora=10.5)
        looper.registrar_passo_arpejo("LA4", "menor", "piano", 1, 0.7, agora=10.8)
        self.assertEqual(looper.contar_eventos(), 3)

        # Fecha gravação -> inicia reprodução
        estado = looper.alternar(11.0)
        self.assertEqual(estado, "tocando")
        self.assertAlmostEqual(looper.duracao, 1.0, places=3)

        # Avança dentro e através do segundo ciclo
        looper.atualizar(11.6)
        self.assertAlmostEqual(looper.obter_progresso(11.6), 0.6, places=2)
        looper.atualizar(12.2)

        # Novo toque para o looper
        self.assertEqual(looper.alternar(12.5), "parado")
        self.assertEqual(looper.contar_eventos(), 0)


class TestMemeOverlay(unittest.TestCase):
    """Testes do sistema probabilístico de partículas e padronização estilo sad_cat_3."""

    def test_assets_sad_padronizados_sem_fundo_quadrado(self) -> None:
        assets = RAIZ_PROJETO / "assets"
        for subpasta in ("happy", "sad"):
            arquivos = sorted((assets / subpasta).glob("*.png"))
            self.assertGreaterEqual(len(arquivos), 3)
            for arq in arquivos:
                img = cv2.imread(str(arq), cv2.IMREAD_UNCHANGED)
                self.assertIsNotNone(img)
                self.assertEqual(img.shape, (256, 256, 4))
                self.assertEqual(int(img[0, 0, 3]), 0)

    def test_disparo_probabilistico_e_cooldown(self) -> None:
        sistema = SistemaParticulasMemes(cooldown_segundos=1.5, probabilidade_meme=0.4)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.assertFalse(sistema.disparar("maior", (0.30, 0.60), agora=10.0, probabilidade=0.0))
        self.assertTrue(sistema.disparar("maior", (0.30, 0.60), rotulo="DO4 MAIOR", agora=10.0, probabilidade=1.0))
        self.assertFalse(sistema.disparar("menor", (0.35, 0.60), rotulo="RE4 MENOR", agora=10.8, probabilidade=1.0))
        self.assertTrue(sistema.disparar("menor", (0.35, 0.60), rotulo="RE4 MENOR", agora=11.6, probabilidade=1.0))
        sistema.atualizar_e_renderizar(frame, agora=11.8)
        self.assertGreater(int(frame.sum()), 0)
        sistema.atualizar_e_renderizar(frame, agora=13.2)
        self.assertEqual(sistema.contar_ativas(), 0)

    def test_curva_animacao(self) -> None:
        e0, a0 = calcular_curva_animacao(0.01)
        ep, _ = calcular_curva_animacao(0.18)
        em, am = calcular_curva_animacao(0.50)
        ef, af = calcular_curva_animacao(0.98)
        self.assertLess(e0, ep)
        self.assertGreater(ep, em)
        self.assertAlmostEqual(am, 1.0, places=3)
        self.assertLess(ef, em)
        self.assertLess(af, 0.1)
        self.assertLess(a0, 0.2)


class TestUIRenderer(unittest.TestCase):
    """Testes da barra superior limpa, corda luminosa, modal [?] e performance HD."""

    def test_renderizacao_tela_limpa_corda_e_performance_720p(self) -> None:
        rastreador = RastreadorMaos()
        esq = _criar_mao_sintetica((True, False, False, False), polegar_aberto=True, lado="esquerda", centro_x=0.35)
        dir_ = _criar_mao_sintetica((True, True, True, True), polegar_aberto=True, lado="direita", centro_x=0.60)
        handed = [_FakeHandedness([_FakeClassificacao("Left")]), _FakeHandedness([_FakeClassificacao("Right")])]
        estado = rastreador.processar([esq, dir_], handed)
        self.assertTrue(estado.arpejo_ativo)

        memes = SistemaParticulasMemes(cooldown_segundos=0.0, probabilidade_meme=1.0)
        memes.disparar("maior", estado.ancora_meme, rotulo="DO5 MAIOR", agora=1.0, forcar=True)

        frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
        for _ in range(3):
            ui_renderer.desenhar_maos(frame, estado.maos_visuais)
            ui_renderer.desenhar_corda_arpejo(frame, estado.ponto_palma_esq, estado.ponto_palma_dir, "maior", "DO5", 0.05, 1.15)
            memes.atualizar_e_renderizar(frame, agora=1.15)
            botoes = ui_renderer.desenhar_hud(frame, estado, "DO5", "violao", "acorde", "tocando", 0.4, False, "timbre:2", 0.6)

        ids = {b.id_acao for b in botoes}
        self.assertEqual(ids, {"timbre:1", "timbre:2", "timbre:3", "timbre:4", "timbre:5", "looper:toggle", "ajuda:toggle"})

        n_frames = 20
        inicio = time.perf_counter()
        for i in range(n_frames):
            t = 1.0 + i * 0.02
            ui_renderer.desenhar_maos(frame, estado.maos_visuais)
            ui_renderer.desenhar_corda_arpejo(frame, estado.ponto_palma_esq, estado.ponto_palma_dir, "maior", "DO5", 0.04, t)
            memes.atualizar_e_renderizar(frame, agora=t)
            ui_renderer.desenhar_hud(frame, estado, "DO5", "violao", "acorde", "tocando", 0.5, False)
        tempo_medio_ms = ((time.perf_counter() - inicio) / n_frames) * 1000.0
        self.assertLess(tempo_medio_ms, 8.0, f"Renderização demorou {tempo_medio_ms:.2f} ms/frame")

        # Também renderiza modal [?] sem erros
        ui_renderer.desenhar_hud(frame, estado, "DO5", "piano", "acorde", "gravando", 1.2, mostrar_guia=True)


if __name__ == "__main__":
    unittest.main()
