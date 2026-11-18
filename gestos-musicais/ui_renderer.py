"""Renderizador de HUD minimalista, corda luminosa e esqueleto de mãos.

Mantém a tela ~92% limpa concentrando os controles em uma barra superior
única com botões interativos (touch aéreo e clique do mouse) para os 5
instrumentos, Mini-Looper e ajuda sob demanda (tecla G ou botão [?]).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Sequence

import cv2
import numpy as np

from audio_engine import NOMES_PADROES_ARPEJO, NOMES_TIMBRES, TIMBRES
from hand_tracker import (
    EstadoMaos,
    INDICADOR_TIP,
    MEDIO_MCP,
    MaoRastreada,
    VOL_MAX,
    VOL_MIN,
)

# Conexões padrão dos 21 landmarks do MediaPipe Hands
CONEXOES_MAO: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20),
)
# Cadeias contínuas equivalentes às 21 conexões para renderização rápida via polylines
_CADEIAS_MAO: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (5, 9, 10, 11, 12),
    (9, 13, 14, 15, 16),
    (13, 17, 18, 19, 20),
    (0, 17),
)
PONTAS_DEDOS = {4, 8, 12, 16, 20}

GUIA_NOTAS_COL_1: tuple[tuple[str, str], ...] = (
    ("DO4", "DO4: 1 Dedo (Ind)"),
    ("RE4", "RE4: 2 Dedos (V)"),
    ("MI4", "MI4: 3 Dedos"),
    ("FA4", "FA4: 4 Dedos (s/Pol)"),
)
GUIA_NOTAS_COL_2: tuple[tuple[str, str], ...] = (
    ("SOL4", "SOL4: 5 Dedos"),
    ("LA4", "LA4: Hang Loose"),
    ("SI4", "SI4: Rock (Ind+Min)"),
    ("DO5", "DO5: Gesto L (Pol+Ind)"),
)


@dataclass(frozen=True)
class BotaoUI:
    """Região clicável/tocável na barra superior da interface."""

    id_acao: str
    rotulo: str
    x1: int
    y1: int
    x2: int
    y2: int

    def contem(self, x: int, y: int) -> bool:
        """Verifica se o ponto ``(x, y)`` está dentro da área do botão."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class FaiscaCorda:
    """Partícula luminosa emitida pelo nó ativo da corda do Arpejador."""

    x: float
    y: float
    vx: float
    vy: float
    cor_bgr: tuple[int, int, int]
    raio: int
    inicio: float
    duracao: float


class SistemaFaiscasCorda:
    """Gerencia faíscas leves emitidas pelos nós da corda do Arpejador."""

    def __init__(self, max_faiscas: int = 48) -> None:
        self.max_faiscas = max_faiscas
        self._faiscas: list[FaiscaCorda] = []

    def contar_ativas(self) -> int:
        """Retorna a quantidade de faíscas atualmente ativas."""
        return len(self._faiscas)

    def emitir(
        self,
        x_px: float,
        y_px: float,
        cor_bgr: tuple[int, int, int],
        quantidade: int = 7,
        agora: float | None = None,
    ) -> None:
        """Dispara ``quantidade`` faíscas radiais a partir do nó tocado."""
        instante = time.perf_counter() if agora is None else agora
        for _ in range(quantidade):
            ang = random.uniform(0.0, math.tau)
            vel = random.uniform(55.0, 145.0)
            self._faiscas.append(
                FaiscaCorda(
                    x=x_px,
                    y=y_px,
                    vx=math.cos(ang) * vel,
                    vy=math.sin(ang) * vel - 25.0,
                    cor_bgr=cor_bgr,
                    raio=random.choice((2, 3, 3)),
                    inicio=instante,
                    duracao=random.uniform(0.28, 0.42),
                )
            )
        if len(self._faiscas) > self.max_faiscas:
            self._faiscas = self._faiscas[-self.max_faiscas :]

    def atualizar_e_renderizar(self, frame: np.ndarray, agora: float | None = None) -> None:
        """Atualiza posição/brilho e desenha as faíscas vigentes sobre o frame."""
        if not self._faiscas:
            return

        instante = time.perf_counter() if agora is None else agora
        h, w = frame.shape[:2]
        restantes: list[FaiscaCorda] = []

        for f in self._faiscas:
            decorrido = instante - f.inicio
            if decorrido < 0.0:
                restantes.append(f)
                continue
            if decorrido >= f.duracao:
                continue

            restantes.append(f)
            progresso = decorrido / f.duracao
            brilho = max(0.0, 1.0 - (progresso ** 1.35))

            px = int(round(f.x + f.vx * decorrido))
            py = int(round(f.y + f.vy * decorrido + 75.0 * (decorrido ** 2)))
            if not (0 <= px < w and 0 <= py < h):
                continue

            cor = (
                int(f.cor_bgr[0] * brilho),
                int(f.cor_bgr[1] * brilho),
                int(f.cor_bgr[2] * brilho),
            )
            raio_atual = max(1, int(round(f.raio * (0.4 + 0.6 * brilho))))
            cv2.circle(frame, (px, py), raio_atual, cor, -1, cv2.LINE_AA)
            if brilho > 0.55:
                cv2.circle(frame, (px, py), 1, (250, 252, 255), -1, cv2.LINE_AA)

        self._faiscas = restantes


_CACHE_BLOCOS_COR: dict[tuple[int, int, int], np.ndarray] = {}


def _obter_bloco_cor(cor_bgr: tuple[int, int, int], altura: int, largura: int) -> np.ndarray:
    """Retorna fatia ``(altura, largura, 3)`` pré-preenchida sem realocar por frame."""
    bloco = _CACHE_BLOCOS_COR.get(cor_bgr)
    if bloco is None or bloco.shape[0] < altura or bloco.shape[1] < largura:
        nova_alt = max(altura, 160 if bloco is None else bloco.shape[0])
        nova_larg = max(largura, 1280 if bloco is None else bloco.shape[1])
        bloco = np.empty((nova_alt, nova_larg, 3), dtype=np.uint8)
        bloco[:, :] = cor_bgr
        _CACHE_BLOCOS_COR[cor_bgr] = bloco
    return bloco[:altura, :largura]


def desenhar_caixa_translucida(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    cor_bgr: tuple[int, int, int],
    alpha: float,
    cor_borda: tuple[int, int, int] | None = None,
    espessura_borda: int = 1,
) -> None:
    """Desenha painel retangular semitransparente operando apenas na ROI."""
    h, w = frame.shape[:2]
    cx1 = max(0, min(w, x1))
    cy1 = max(0, min(h, y1))
    cx2 = max(0, min(w, x2))
    cy2 = max(0, min(h, y2))
    if cx2 <= cx1 or cy2 <= cy1:
        return

    roi = frame[cy1:cy2, cx1:cx2]
    a = max(0.0, min(1.0, float(alpha)))
    fundo = _obter_bloco_cor(cor_bgr, cy2 - cy1, cx2 - cx1)
    cv2.addWeighted(fundo, a, roi, 1.0 - a, 0.0, dst=roi)

    if cor_borda is not None and espessura_borda > 0:
        cv2.rectangle(frame, (cx1, cy1), (cx2 - 1, cy2 - 1), cor_borda, espessura_borda, cv2.LINE_AA)


def _cor_nivel_volume(nivel: float) -> tuple[int, int, int]:
    """Interpola cor BGR verde -> âmbar -> coral conforme o nível de volume."""
    progresso = max(0.0, min(1.0, (nivel - VOL_MIN) / (VOL_MAX - VOL_MIN)))
    if progresso < 0.5:
        t = progresso * 2.0
        b = int(95 * (1.0 - t) + 45 * t)
        g = int(220 * (1.0 - t) + 215 * t)
        r = int(75 * (1.0 - t) + 250 * t)
    else:
        t = (progresso - 0.5) * 2.0
        b = int(45 * (1.0 - t) + 75 * t)
        g = int(215 * (1.0 - t) + 115 * t)
        r = int(250 * (1.0 - t) + 255 * t)
    return (b, g, r)


def _cores_mao(mao: MaoRastreada) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Define cor principal e cor de halo suave do esqueleto conforme o papel."""
    if mao.papel == "esquerda":
        if mao.nota is not None:
            return (235, 220, 75), (115, 105, 35)
        return (165, 175, 190), (80, 88, 100)

    if mao.papel == "direita":
        if mao.postura_direita == "aberta":
            return (95, 240, 145), (45, 120, 70)
        return (255, 195, 80), (130, 95, 35)

    return (145, 235, 95), (70, 115, 45)


def desenhar_maos(
    frame: np.ndarray,
    maos_visuais: Sequence[MaoRastreada],
) -> None:
    """Desenha esqueletos limpos e suavizados das mãos, com cursor aéreo no topo."""
    if not maos_visuais:
        return

    h, w = frame.shape[:2]
    for mao in maos_visuais:
        pts_px = np.empty((21, 2), dtype=np.int32)
        pts_px[:, 0] = np.clip(np.round(mao.pontos_norm[:, 0] * w), 0, w - 1).astype(np.int32)
        pts_px[:, 1] = np.clip(np.round(mao.pontos_norm[:, 1] * h), 0, h - 1).astype(np.int32)

        cor_principal, cor_halo = _cores_mao(mao)

        pad = 12
        bx1 = max(0, int(pts_px[:, 0].min()) - pad)
        by1 = max(0, int(pts_px[:, 1].min()) - pad)
        bx2 = min(w, int(pts_px[:, 0].max()) + pad + 1)
        by2 = min(h, int(pts_px[:, 1].max()) + pad + 1)
        if bx2 > bx1 and by2 > by1:
            roi = frame[by1:by2, bx1:bx2]
            camada_halo = roi.copy()
            pts_roi_base = pts_px - np.array([bx1, by1], dtype=np.int32)
            cadeias_roi = [pts_roi_base[list(cadeia)] for cadeia in _CADEIAS_MAO]
            cv2.polylines(camada_halo, cadeias_roi, False, cor_halo, 5, cv2.LINE_AA)
            cv2.addWeighted(camada_halo, 0.36, roi, 0.64, 0.0, dst=roi)

        cadeias_px = [pts_px[list(cadeia)] for cadeia in _CADEIAS_MAO]
        cv2.polylines(frame, cadeias_px, False, cor_principal, 2, cv2.LINE_AA)

        for idx in range(21):
            centro = (int(pts_px[idx, 0]), int(pts_px[idx, 1]))
            if idx in PONTAS_DEDOS:
                cv2.circle(frame, centro, 5, cor_principal, -1, cv2.LINE_AA)
                cv2.circle(frame, centro, 2, (248, 250, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, centro, 3, cor_principal, -1, cv2.LINE_AA)

        # Indicador discreto no centro da palma
        centro_palma = (int(pts_px[MEDIO_MCP, 0]), int(pts_px[MEDIO_MCP, 1]))
        cv2.circle(frame, centro_palma, 6, cor_principal, 1, cv2.LINE_AA)

        # Cursor luminoso quando a ponta do indicador entra na barra superior (Touch Aéreo)
        if float(mao.pontos_norm[INDICADOR_TIP, 1]) < 0.12:
            ponta = (int(pts_px[INDICADOR_TIP, 0]), int(pts_px[INDICADOR_TIP, 1]))
            cv2.circle(frame, ponta, 10, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, ponta, 5, (80, 235, 255), -1, cv2.LINE_AA)


def desenhar_corda_arpejo(
    frame: np.ndarray,
    ponto_palma_esq: tuple[float, float] | None,
    ponto_palma_dir: tuple[float, float] | None,
    escala: str,
    nota_ativa: str | None,
    tempo_desde_pulso: float,
    agora: float,
    grau_ativo: int = 0,
) -> list[tuple[int, int]]:
    """Desenha corda luminosa com 4 nós (Tônica, 3ª, 5ª, 8ª) e retorna suas posições."""
    if ponto_palma_esq is None or ponto_palma_dir is None:
        return []

    h, w = frame.shape[:2]
    x1 = ponto_palma_esq[0] * w
    y1 = ponto_palma_esq[1] * h
    x2 = ponto_palma_dir[0] * w
    y2 = ponto_palma_dir[1] * h

    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 10.0:
        return []

    nx = -dy / dist
    ny = dx / dist

    if nota_ativa is not None:
        decaimento = math.exp(-max(0.0, tempo_desde_pulso) * 8.5)
        amplitude = 2.5 + 18.0 * decaimento
        cor_glow = (70, 215, 130) if escala == "maior" else (245, 165, 65)
        cor_nucleo = (145, 250, 190) if escala == "maior" else (255, 215, 125)
    else:
        amplitude = 1.5
        cor_glow = (100, 120, 145)
        cor_nucleo = (180, 200, 225)

    # 31 pontos -> índices 6, 12, 18, 24 correspondem a u = 0.20, 0.40, 0.60, 0.80
    n_pts = 31
    fase = agora * 28.0
    pts = np.empty((n_pts, 2), dtype=np.int32)
    for i in range(n_pts):
        u = i / (n_pts - 1)
        envelope = math.sin(math.pi * u)
        onda = math.sin(3.0 * math.pi * u + fase) * envelope * amplitude
        px = int(round(x1 + dx * u + nx * onda))
        py = int(round(y1 + dy * u + ny * onda))
        pts[i, 0] = max(0, min(w - 1, px))
        pts[i, 1] = max(0, min(h - 1, py))

    pad = 14
    bx1 = max(0, int(pts[:, 0].min()) - pad)
    by1 = max(0, int(pts[:, 1].min()) - pad)
    bx2 = min(w, int(pts[:, 0].max()) + pad + 1)
    by2 = min(h, int(pts[:, 1].max()) + pad + 1)
    if bx2 > bx1 and by2 > by1:
        roi = frame[by1:by2, bx1:bx2]
        camada = roi.copy()
        pts_roi = pts - np.array([bx1, by1], dtype=np.int32)
        cv2.polylines(camada, [pts_roi], False, cor_glow, 7 if nota_ativa else 3, cv2.LINE_AA)
        cv2.addWeighted(camada, 0.45, roi, 0.55, 0.0, dst=roi)

    cv2.polylines(frame, [pts], False, cor_nucleo, 3 if nota_ativa else 1, cv2.LINE_AA)
    if nota_ativa is not None:
        cv2.polylines(frame, [pts], False, (248, 252, 255), 1, cv2.LINE_AA)

    # 4 nós representando Tônica (0), Terça (1), Quinta (2) e Oitava (3)
    indices_nos = (6, 12, 18, 24)
    posicoes_nos: list[tuple[int, int]] = []
    grau_norm = grau_ativo % 4
    for g, idx_pt in enumerate(indices_nos):
        nx_px = int(pts[idx_pt, 0])
        ny_px = int(pts[idx_pt, 1])
        posicoes_nos.append((nx_px, ny_px))

        if nota_ativa is not None and g == grau_norm:
            cv2.circle(frame, (nx_px, ny_px), 11, cor_glow, 2, cv2.LINE_AA)
            cv2.circle(frame, (nx_px, ny_px), 7, cor_nucleo, -1, cv2.LINE_AA)
            cv2.circle(frame, (nx_px, ny_px), 3, (252, 254, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (nx_px, ny_px), 4, cor_nucleo, -1, cv2.LINE_AA)
            cv2.circle(frame, (nx_px, ny_px), 5, (30, 38, 52), 1, cv2.LINE_AA)

    return posicoes_nos


def desenhar_guia_gestos(
    frame: np.ndarray,
    nota_ativa: str | None,
    escala: str,
) -> None:
    """Desenha card compacto de ajuda/gestos apenas quando ativado (G ou [?])."""
    altura, largura = frame.shape[:2]
    fonte = cv2.FONT_HERSHEY_SIMPLEX

    gx1 = 16
    gy1 = 72
    gx2 = min(largura - 16, gx1 + 528)
    gy2 = min(altura - 16, gy1 + 148)
    if gx2 - gx1 < 280 or gy2 - gy1 < 120:
        return

    desenhar_caixa_translucida(
        frame,
        gx1,
        gy1,
        gx2,
        gy2,
        (14, 18, 28),
        0.82,
        cor_borda=(82, 98, 132),
        espessura_borda=1,
    )

    cor_destaque = (95, 245, 150) if escala == "maior" else (255, 205, 105)
    cv2.putText(
        frame,
        "GUIA DE GESTOS E ATALHOS",
        (gx1 + 12, gy1 + 20),
        fonte,
        0.44,
        (242, 246, 252),
        1,
        cv2.LINE_AA,
    )

    y_topo = gy1 + 40
    passo_y = 18
    col1_x = gx1 + 12
    col2_x = gx1 + 246

    for col_x, itens in ((col1_x, GUIA_NOTAS_COL_1), (col2_x, GUIA_NOTAS_COL_2)):
        for idx, (codigo, texto) in enumerate(itens):
            cy = y_topo + idx * passo_y
            ativo = nota_ativa == codigo
            if ativo:
                cv2.circle(frame, (col_x + 4, cy - 4), 3, cor_destaque, -1, cv2.LINE_AA)
                cor_item = (255, 255, 255)
            else:
                cor_item = (182, 194, 214)
            cv2.putText(
                frame,
                texto,
                (col_x + (12 if ativo else 0), cy),
                fonte,
                0.38,
                cor_item,
                1,
                cv2.LINE_AA,
            )

    cv2.line(frame, (gx1 + 10, gy2 - 42), (gx2 - 10, gy2 - 42), (55, 66, 92), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "Esq Punho: Silencio | Dir Aberta: MAIOR / Fechada: MENOR | Juntas: Arpejo",
        (gx1 + 12, gy2 - 24),
        fonte,
        0.35,
        (195, 210, 232),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "1-5: Inst | A: Arpejo | R: Loop | H: Acorde/Nota | F: Tela Cheia | G: Fechar | Q: Sair",
        (gx1 + 12, gy2 - 8),
        fonte,
        0.33,
        (165, 182, 208),
        1,
        cv2.LINE_AA,
    )


def desenhar_hud(
    frame: np.ndarray,
    estado_maos: EstadoMaos,
    nota_ativa: str | None,
    timbre: str,
    modo_harmonico: str,
    looper_estado: str = "parado",
    looper_progresso: float = 0.0,
    mostrar_guia: bool = False,
    botao_hover_id: str | None = None,
    progresso_hover: float = 0.0,
    padrao_arpejo: str = "sobe",
) -> list[BotaoUI]:
    """Renderiza a barra superior única minimalista e retorna seus botões interativos."""
    _, largura = frame.shape[:2]
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    botoes: list[BotaoUI] = []

    p_x1, p_y1 = 12, 8
    p_x2, p_y2 = largura - 12, 62
    desenhar_caixa_translucida(
        frame,
        p_x1,
        p_y1,
        p_x2,
        p_y2,
        (14, 18, 28),
        0.74,
        cor_borda=(68, 80, 108),
        espessura_borda=1,
    )

    # 1. Bloco Esquerdo: Modo (1M/2M), Nota Ativa, Pill MAIOR/MENOR, Volume e Botão ARP
    if estado_maos.num_maos == 0:
        cor_led = (45, 175, 255)
        txt_modo = "--"
    elif estado_maos.modo_duas_maos:
        cor_led = (95, 240, 145)
        txt_modo = "2M"
    else:
        cor_led = (145, 235, 95)
        txt_modo = "1M"

    cv2.circle(frame, (26, 35), 5, cor_led, -1, cv2.LINE_AA)
    cv2.putText(frame, txt_modo, (36, 40), fonte, 0.45, (215, 224, 240), 1, cv2.LINE_AA)

    eh_maior = estado_maos.escala == "maior"
    cor_nota = (95, 245, 150) if eh_maior else (255, 205, 105)
    txt_nota = nota_ativa if nota_ativa is not None else "---"
    if nota_ativa is None:
        cor_nota = (145, 156, 176)
    cv2.putText(frame, txt_nota, (66, 43), fonte, 0.76, cor_nota, 2, cv2.LINE_AA)

    # Pill MAIOR (verde quente) / MENOR (azul frio)
    pill_x1, pill_y1 = 142, 18
    pill_x2, pill_y2 = 214, 52
    cor_pill_f = (30, 92, 54) if eh_maior else (95, 52, 22)
    cor_pill_b = (95, 240, 145) if eh_maior else (255, 195, 80)
    desenhar_caixa_translucida(frame, pill_x1, pill_y1, pill_x2, pill_y2, cor_pill_f, 0.85, cor_pill_b, 1)
    txt_esc = "MAIOR" if eh_maior else "MENOR"
    (ew, eh), _ = cv2.getTextSize(txt_esc, fonte, 0.45, 2)
    cv2.putText(
        frame,
        txt_esc,
        (pill_x1 + ((pill_x2 - pill_x1) - ew) // 2, pill_y1 + ((pill_y2 - pill_y1) + eh) // 2),
        fonte,
        0.45,
        (248, 252, 255),
        2,
        cv2.LINE_AA,
    )

    # Mini-barra horizontal de volume compacta
    vx1, vy1, vx2, vy2 = 222, 26, 274, 34
    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (40, 46, 62), -1)
    prog_vol = max(0.0, min(1.0, (estado_maos.volume - VOL_MIN) / (VOL_MAX - VOL_MIN)))
    preench = max(2, int(round((vx2 - vx1) * prog_vol)))
    cor_vol = _cor_nivel_volume(estado_maos.volume)
    cv2.rectangle(frame, (vx1, vy1), (vx1 + preench, vy2), cor_vol, -1)
    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (110, 122, 148), 1, cv2.LINE_AA)
    txt_vol = f"{int(round(estado_maos.volume * 100))}% {'A' if modo_harmonico == 'acorde' else 'N'}"
    cv2.putText(frame, txt_vol, (222, 51), fonte, 0.35, cor_vol, 1, cv2.LINE_AA)

    # Botão interativo ARP:<PADRAO> (tecla A, Touch Aéreo ou clique)
    arp_x1, arp_y1, arp_x2, arp_y2 = 284, 16, 384, 54
    nome_padrao = NOMES_PADROES_ARPEJO.get(padrao_arpejo, padrao_arpejo).upper()
    rotulo_arp = f"ARP:{nome_padrao}"
    btn_arp = BotaoUI("arpejo:padrao", rotulo_arp, arp_x1, arp_y1, arp_x2, arp_y2)
    botoes.append(btn_arp)

    arp_f = (25, 86, 108) if estado_maos.arpejo_ativo else (24, 30, 44)
    arp_b = (85, 230, 255) if estado_maos.arpejo_ativo else (78, 92, 120)
    desenhar_caixa_translucida(frame, arp_x1, arp_y1, arp_x2, arp_y2, arp_f, 0.85, arp_b, 2 if estado_maos.arpejo_ativo else 1)
    (aw, ah), _ = cv2.getTextSize(rotulo_arp, fonte, 0.38, 1)
    cv2.putText(
        frame,
        rotulo_arp,
        (arp_x1 + max(4, ((arp_x2 - arp_x1) - aw) // 2), arp_y1 + ((arp_y2 - arp_y1) + ah) // 2),
        fonte,
        0.38,
        (248, 252, 255) if estado_maos.arpejo_ativo else (188, 200, 220),
        1,
        cv2.LINE_AA,
    )
    if botao_hover_id == btn_arp.id_acao and progresso_hover > 0.0:
        hw = int(round((arp_x2 - arp_x1 - 4) * min(1.0, progresso_hover)))
        cv2.rectangle(frame, (arp_x1 + 2, arp_y2 - 4), (arp_x1 + 2 + hw, arp_y2 - 1), (80, 235, 255), -1)

    cv2.line(frame, (392, 16), (392, 54), (58, 68, 92), 1, cv2.LINE_AA)

    # 2. Botões à Direita: Looper e Ajuda [?]
    aj_x2 = largura - 20
    aj_x1 = aj_x2 - 38
    aj_y1, aj_y2 = 16, 54
    btn_ajuda = BotaoUI("ajuda:toggle", "?", aj_x1, aj_y1, aj_x2, aj_y2)
    botoes.append(btn_ajuda)

    lp_x2 = aj_x1 - 10
    lp_x1 = lp_x2 - 122
    btn_looper = BotaoUI("looper:toggle", "LOOP", lp_x1, aj_y1, lp_x2, aj_y2)
    botoes.append(btn_looper)

    # Desenho do botão do Looper
    if looper_estado == "gravando":
        lp_f, lp_b = (32, 32, 125), (95, 95, 255)
        lp_txt = f"REC {looper_progresso:.1f}s"
    elif looper_estado == "tocando":
        lp_f, lp_b = (32, 92, 68), (95, 240, 175)
        lp_txt = "LOOP ON"
    else:
        lp_f, lp_b = (26, 32, 46), (86, 98, 126)
        lp_txt = "R: LOOP"

    desenhar_caixa_translucida(frame, lp_x1, aj_y1, lp_x2, aj_y2, lp_f, 0.85, lp_b, 1)
    (lw, lh), _ = cv2.getTextSize(lp_txt, fonte, 0.43, 1)
    cv2.putText(
        frame,
        lp_txt,
        (lp_x1 + ((lp_x2 - lp_x1) - lw) // 2, aj_y1 + ((aj_y2 - aj_y1) + lh) // 2 - 2),
        fonte,
        0.43,
        (246, 250, 255),
        1,
        cv2.LINE_AA,
    )
    if looper_estado == "tocando":
        prog_w = int(round((lp_x2 - lp_x1 - 6) * max(0.0, min(1.0, looper_progresso))))
        if prog_w > 0:
            cv2.rectangle(frame, (lp_x1 + 3, aj_y2 - 5), (lp_x1 + 3 + prog_w, aj_y2 - 2), (95, 245, 180), -1)
    if botao_hover_id == btn_looper.id_acao and progresso_hover > 0.0:
        hw = int(round((lp_x2 - lp_x1 - 4) * min(1.0, progresso_hover)))
        cv2.rectangle(frame, (lp_x1 + 2, aj_y2 - 4), (lp_x1 + 2 + hw, aj_y2 - 1), (80, 235, 255), -1)

    # Desenho do botão [?]
    aj_f = (44, 78, 122) if mostrar_guia else (26, 32, 46)
    aj_b = (115, 215, 255) if mostrar_guia else (86, 98, 126)
    desenhar_caixa_translucida(frame, aj_x1, aj_y1, aj_x2, aj_y2, aj_f, 0.85, aj_b, 1)
    (qw, qh), _ = cv2.getTextSize("?", fonte, 0.56, 2)
    cv2.putText(
        frame,
        "?",
        (aj_x1 + ((aj_x2 - aj_x1) - qw) // 2, aj_y1 + ((aj_y2 - aj_y1) + qh) // 2),
        fonte,
        0.56,
        (248, 252, 255),
        2,
        cv2.LINE_AA,
    )
    if botao_hover_id == btn_ajuda.id_acao and progresso_hover > 0.0:
        hw = int(round((aj_x2 - aj_x1 - 4) * min(1.0, progresso_hover)))
        cv2.rectangle(frame, (aj_x1 + 2, aj_y2 - 4), (aj_x1 + 2 + hw, aj_y2 - 1), (80, 235, 255), -1)

    # 3. Seletor Central de 5 Instrumentos (touch aéreo + clique + teclas 1..5)
    area_ini = 400
    area_fim = lp_x1 - 14
    cv2.line(frame, (lp_x1 - 7, 16), (lp_x1 - 7, 54), (58, 68, 92), 1, cv2.LINE_AA)

    n_inst = len(TIMBRES)
    espaco = 8
    larg_btn = max(58, (area_fim - area_ini - (n_inst - 1) * espaco) // n_inst)

    for idx, chave_timbre in enumerate(TIMBRES):
        bx1 = area_ini + idx * (larg_btn + espaco)
        bx2 = min(area_fim, bx1 + larg_btn)
        by1, by2 = 16, 54
        ativo = timbre == chave_timbre
        rotulo = f"{idx + 1} {NOMES_TIMBRES[chave_timbre]}"

        btn = BotaoUI(f"timbre:{idx + 1}", rotulo, bx1, by1, bx2, by2)
        botoes.append(btn)

        fundo_b = (42, 82, 118) if ativo else (24, 30, 44)
        borda_b = (105, 225, 255) if ativo else (72, 84, 112)
        desenhar_caixa_translucida(frame, bx1, by1, bx2, by2, fundo_b, 0.85, borda_b, 2 if ativo else 1)

        escala_txt = 0.42 if larg_btn >= 95 else 0.35
        (tw, th), _ = cv2.getTextSize(rotulo, fonte, escala_txt, 1)
        tx = bx1 + max(4, ((bx2 - bx1) - tw) // 2)
        ty = by1 + ((by2 - by1) + th) // 2
        cv2.putText(
            frame,
            rotulo,
            (tx, ty),
            fonte,
            escala_txt,
            (250, 252, 255) if ativo else (188, 198, 218),
            1,
            cv2.LINE_AA,
        )

        if botao_hover_id == btn.id_acao and progresso_hover > 0.0:
            hw = int(round((bx2 - bx1 - 4) * min(1.0, progresso_hover)))
            cv2.rectangle(frame, (bx1 + 2, by2 - 4), (bx1 + 2 + hw, by2 - 1), (80, 235, 255), -1)

    if mostrar_guia:
        desenhar_guia_gestos(frame, nota_ativa, estado_maos.escala)

    return botoes
