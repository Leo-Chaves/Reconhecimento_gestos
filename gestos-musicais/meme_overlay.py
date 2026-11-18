"""Sistema dinâmico de partículas pop-up de memes (gatos felizes e tristes).

Carrega sprites de ``assets/happy/`` (Modo Maior) e ``assets/sad/`` (Modo Menor)
preservando transparência via canal Alpha (BGRA) e renderiza pop-ups animados
com disparo probabilístico e cooldown mínimo para evitar poluição visual.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
import time
from typing import Literal

import cv2
import numpy as np

CategoriaHumor = Literal["happy", "sad"]
EXTENSOES_IMAGEM = {".png", ".jpg", ".jpeg", ".webp"}

# ---------------------------------------------------------------------------
# Parâmetros de Calibração dos Memes
# ---------------------------------------------------------------------------
# Probabilidade (0.0 a 1.0) de um meme aparecer quando uma nova nota é atacada.
PROBABILIDADE_MEME: float = 0.40
# Intervalo mínimo em segundos entre dois disparos para não poluir a tela.
COOLDOWN_MEME_SEGUNDOS: float = 1.5
TAMANHO_BASE_SPRITE: int = 136
MAX_PARTICULAS_ATIVAS: int = 6


@dataclass
class ParticulaMeme:
    """Representa um pop-up animado em execução na tela."""

    categoria: CategoriaHumor
    imagem_bgra: np.ndarray
    rotulo: str
    ancora_x_norm: float
    ancora_y_norm: float
    offset_x_px: float
    offset_y_px: float
    vel_x_px: float
    angulo_inicial_graus: float
    vel_angular_graus: float
    fase_lateral: float
    inicio: float
    duracao: float


def _redimensionar_proporcional(imagem: np.ndarray, lado_max: int) -> np.ndarray:
    """Redimensiona a imagem mantendo proporção dentro de ``lado_max x lado_max``."""
    h, w = imagem.shape[:2]
    if h <= 0 or w <= 0:
        return imagem
    escala = min(lado_max / float(w), lado_max / float(h))
    novo_w = max(8, int(round(w * escala)))
    novo_h = max(8, int(round(h * escala)))
    interp = cv2.INTER_AREA if escala < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(imagem, (novo_w, novo_h), interpolation=interp)


def _garantir_bgra(imagem: np.ndarray) -> np.ndarray:
    """Garante que a imagem possua 4 canais (BGRA)."""
    if imagem.ndim == 2:
        bgr = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
        alpha = np.full((bgr.shape[0], bgr.shape[1], 1), 255, dtype=np.uint8)
        return np.concatenate((bgr, alpha), axis=2)
    if imagem.shape[2] == 3:
        alpha = np.full((imagem.shape[0], imagem.shape[1], 1), 255, dtype=np.uint8)
        return np.concatenate((imagem, alpha), axis=2)
    if imagem.shape[2] == 4:
        return imagem
    return imagem[:, :, :4]


def _rotacionar_bgra_sem_corte(imagem_bgra: np.ndarray, angulo_graus: float) -> np.ndarray:
    """Rotaciona o sprite BGRA expandindo o recorte para não cortar bordas."""
    if abs(angulo_graus) < 0.25:
        return imagem_bgra
    h, w = imagem_bgra.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    matriz = cv2.getRotationMatrix2D((cx, cy), angulo_graus, 1.0)
    cos_a = abs(matriz[0, 0])
    sin_a = abs(matriz[0, 1])
    novo_w = max(8, int(math.ceil(h * sin_a + w * cos_a)))
    novo_h = max(8, int(math.ceil(h * cos_a + w * sin_a)))
    matriz[0, 2] += (novo_w * 0.5) - cx
    matriz[1, 2] += (novo_h * 0.5) - cy
    return cv2.warpAffine(
        imagem_bgra,
        matriz,
        (novo_w, novo_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _criar_fallback_gato_bgra(categoria: CategoriaHumor) -> np.ndarray:
    """Gera sprite BGRA vetorial de emergência caso a pasta de assets esteja vazia."""
    lado = TAMANHO_BASE_SPRITE
    canvas = np.zeros((lado, lado, 4), dtype=np.uint8)
    cx, cy = lado // 2, int(lado * 0.56)
    raio = int(lado * 0.34)
    cor_pelo = (120, 195, 250, 255) if categoria == "happy" else (215, 200, 185, 255)
    borda = (45, 40, 50, 255)

    orelha_e = np.array([[cx - raio + 6, cy - 10], [cx - raio + 2, cy - raio - 16], [cx - 10, cy - raio + 8]])
    orelha_d = np.array([[cx + raio - 6, cy - 10], [cx + raio - 2, cy - raio - 16], [cx + 10, cy - raio + 8]])
    cv2.fillPoly(canvas, [orelha_e, orelha_d], cor_pelo, cv2.LINE_AA)
    cv2.polylines(canvas, [orelha_e, orelha_d], True, borda, 2, cv2.LINE_AA)

    cv2.circle(canvas, (cx, cy), raio, cor_pelo, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), raio, borda, 2, cv2.LINE_AA)

    for sinal in (-1, 1):
        ox = cx + sinal * 15
        oy = cy - 6
        cv2.circle(canvas, (ox, oy), 5, borda, -1, cv2.LINE_AA)
        if categoria == "sad":
            cv2.ellipse(canvas, (ox, oy + 12), (3, 7), 0, 0, 360, (255, 195, 100, 240), -1, cv2.LINE_AA)

    if categoria == "happy":
        cv2.ellipse(canvas, (cx, cy + 10), (14, 9), 0, 10, 170, borda, 2, cv2.LINE_AA)
    else:
        cv2.ellipse(canvas, (cx, cy + 18), (12, 8), 0, 195, 345, borda, 2, cv2.LINE_AA)
    return canvas


def _sobrepor_bgra_roi(
    destino: np.ndarray,
    origem_bgra: np.ndarray,
    x: int,
    y: int,
    alpha_global: float,
) -> None:
    """Compõe ``origem_bgra`` sobre ``destino`` usando exclusivamente o recorte ROI."""
    if alpha_global <= 0.01:
        return

    h_dst, w_dst = destino.shape[:2]
    h_src, w_src = origem_bgra.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_dst, x + w_src)
    y2 = min(h_dst, y + h_src)
    if x1 >= x2 or y1 >= y2:
        return

    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    recorte_src = origem_bgra[sy1:sy2, sx1:sx2]
    recorte_dst = destino[y1:y2, x1:x2]

    a_glob = max(0.0, min(1.0, float(alpha_global)))
    alpha_u16 = (recorte_src[:, :, 3:4].astype(np.uint16) * int(round(a_glob * 256))) >> 8
    inv_u16 = 255 - alpha_u16
    bgr_src = recorte_src[:, :, :3].astype(np.uint16)
    bgr_dst = recorte_dst.astype(np.uint16)

    mistura = (bgr_src * alpha_u16 + bgr_dst * inv_u16 + 127) // 255
    destino[y1:y2, x1:x2] = mistura.astype(np.uint8)


def _desenhar_tag_translucida(
    frame: np.ndarray,
    texto: str,
    centro_x: int,
    topo_y: int,
    cor_destaque: tuple[int, int, int],
    alpha: float,
) -> None:
    """Desenha etiqueta compacta com alpha blending restrito ao ROI abaixo do meme."""
    if not texto or alpha <= 0.05:
        return

    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala_txt = 0.44
    (tw, th), baseline = cv2.getTextSize(texto, fonte, escala_txt, 1)
    pad_x, pad_y = 9, 4
    larg = tw + pad_x * 2
    alt = th + baseline + pad_y * 2

    h_f, w_f = frame.shape[:2]
    x1 = max(4, min(w_f - larg - 4, centro_x - larg // 2))
    y1 = max(4, min(h_f - alt - 4, topo_y))
    x2 = x1 + larg
    y2 = y1 + alt
    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    fundo = np.full_like(roi, (18, 22, 32))
    cv2.rectangle(fundo, (0, 0), (larg - 1, alt - 1), cor_destaque, 1, cv2.LINE_AA)
    cv2.putText(
        fundo,
        texto,
        (pad_x, pad_y + th),
        fonte,
        escala_txt,
        (242, 246, 252),
        1,
        cv2.LINE_AA,
    )
    peso = 0.85 * alpha
    cv2.addWeighted(fundo, peso, roi, 1.0 - peso, 0.0, dst=roi)


def calcular_curva_animacao(progresso: float) -> tuple[float, float]:
    """Retorna ``(escala, opacidade)`` para um progresso normalizado ``[0.0, 1.0]``."""
    p = max(0.0, min(1.0, progresso))

    if p < 0.18:
        t = p / 0.18
        ease = 1.0 - (1.0 - t) ** 2
        escala = 0.32 + 0.80 * ease
    elif p < 0.28:
        t = (p - 0.18) / 0.10
        escala = 1.12 - 0.12 * t
    elif p < 0.68:
        escala = 1.0
    else:
        t = (p - 0.68) / 0.32
        escala = 1.0 - 0.32 * (t ** 1.2)

    if p < 0.14:
        alpha = p / 0.14
    elif p < 0.64:
        alpha = 1.0
    else:
        t = (p - 0.64) / 0.36
        alpha = max(0.0, (1.0 - t) ** 1.35)

    return escala, alpha


class SistemaParticulasMemes:
    """Gerencia pop-ups probabilísticos de gatos com rotação, deriva e canal Alpha."""

    def __init__(
        self,
        diretorio_assets: Path | None = None,
        cooldown_segundos: float = COOLDOWN_MEME_SEGUNDOS,
        probabilidade_meme: float = PROBABILIDADE_MEME,
    ) -> None:
        base = diretorio_assets or (Path(__file__).resolve().parent / "assets")
        self.diretorio_assets = base
        self.cooldown_segundos = max(0.0, float(cooldown_segundos))
        self.probabilidade_meme = max(0.0, min(1.0, float(probabilidade_meme)))

        self._sprites: dict[CategoriaHumor, list[np.ndarray]] = {
            "happy": self._carregar_pasta(base / "happy", "happy"),
            "sad": self._carregar_pasta(base / "sad", "sad"),
        }
        self._indices: dict[CategoriaHumor, int] = {"happy": 0, "sad": 0}
        self._particulas: list[ParticulaMeme] = []
        self._ultimo_disparo: float = -999.0

    @staticmethod
    def _carregar_pasta(pasta: Path, categoria: CategoriaHumor) -> list[np.ndarray]:
        imagens: list[np.ndarray] = []
        if pasta.exists():
            for arquivo in sorted(pasta.iterdir()):
                if not arquivo.is_file() or arquivo.suffix.lower() not in EXTENSOES_IMAGEM:
                    continue
                img = cv2.imread(str(arquivo), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                bgra = _garantir_bgra(img)
                imagens.append(_redimensionar_proporcional(bgra, TAMANHO_BASE_SPRITE))
        if not imagens:
            imagens.append(_criar_fallback_gato_bgra(categoria))
        return imagens

    def contar_imagens(self) -> dict[CategoriaHumor, int]:
        """Retorna a quantidade de sprites carregados por pasta (``happy``/``sad``)."""
        return {cat: len(imgs) for cat, imgs in self._sprites.items()}

    def contar_ativas(self) -> int:
        """Retorna o número atual de partículas vivas na tela."""
        return len(self._particulas)

    def _escolher_sprite(self, categoria: CategoriaHumor) -> np.ndarray:
        lista = self._sprites[categoria]
        if len(lista) == 1:
            return lista[0]
        idx = self._indices[categoria] % len(lista)
        self._indices[categoria] += 1
        if random.random() < 0.35:
            idx = random.randrange(len(lista))
        return lista[idx]

    def disparar(
        self,
        escala_ou_humor: str,
        posicao_norm: tuple[float, float] | None = None,
        rotulo: str = "",
        agora: float | None = None,
        forcar: bool = False,
        probabilidade: float | None = None,
    ) -> bool:
        """Dispara um pop-up de gato feliz/triste respeitando cooldown e probabilidade."""
        instante = time.perf_counter() if agora is None else agora

        if not forcar:
            if (instante - self._ultimo_disparo) < self.cooldown_segundos:
                return False
            prob = self.probabilidade_meme if probabilidade is None else max(0.0, min(1.0, float(probabilidade)))
            if random.random() >= prob:
                return False

        chave = escala_ou_humor.lower()
        categoria: CategoriaHumor = "happy" if chave in ("maior", "happy", "good") else "sad"
        sprite = self._escolher_sprite(categoria)

        ax, ay = posicao_norm if posicao_norm is not None else (0.35, 0.56)
        ax = max(0.14, min(0.76, float(ax)))
        ay = max(0.24, min(0.84, float(ay)))

        particula = ParticulaMeme(
            categoria=categoria,
            imagem_bgra=sprite,
            rotulo=rotulo,
            ancora_x_norm=ax,
            ancora_y_norm=ay,
            offset_x_px=random.uniform(-34.0, 34.0),
            offset_y_px=random.uniform(-76.0, -34.0),
            vel_x_px=random.uniform(-26.0, 26.0),
            angulo_inicial_graus=random.uniform(-12.0, 12.0),
            vel_angular_graus=random.uniform(-12.0, 12.0),
            fase_lateral=random.uniform(0.0, math.tau),
            inicio=instante,
            duracao=random.uniform(0.90, 1.15),
        )
        if len(self._particulas) >= MAX_PARTICULAS_ATIVAS:
            self._particulas.pop(0)
        self._particulas.append(particula)
        self._ultimo_disparo = instante
        return True

    def atualizar_e_renderizar(self, frame: np.ndarray, agora: float | None = None) -> None:
        """Atualiza física, rotação, transparência e renderiza as partículas vigentes."""
        if not self._particulas:
            return

        instante = time.perf_counter() if agora is None else agora
        h_frame, w_frame = frame.shape[:2]
        restantes: list[ParticulaMeme] = []

        for p in self._particulas:
            decorrido = instante - p.inicio
            if decorrido < 0.0:
                restantes.append(p)
                continue
            if decorrido >= p.duracao:
                continue

            restantes.append(p)
            progresso = decorrido / p.duracao
            escala, alpha = calcular_curva_animacao(progresso)

            if p.categoria == "happy":
                desloc_y = -68.0 * decorrido
                desloc_x = p.vel_x_px * decorrido + math.sin(decorrido * 6.5 + p.fase_lateral) * 6.0
                cor_tag = (95, 235, 145)
            else:
                desloc_y = -32.0 * decorrido
                desloc_x = p.vel_x_px * decorrido + math.sin(decorrido * 4.5 + p.fase_lateral) * 4.0
                cor_tag = (250, 190, 90)

            h_src, w_src = p.imagem_bgra.shape[:2]
            w_esc = max(16, int(round(w_src * escala)))
            h_esc = max(16, int(round(h_src * escala)))
            interp = cv2.INTER_AREA if escala < 1.0 else cv2.INTER_LINEAR
            sprite_esc = cv2.resize(p.imagem_bgra, (w_esc, h_esc), interpolation=interp)

            angulo_atual = p.angulo_inicial_graus + p.vel_angular_graus * decorrido
            sprite_anim = _rotacionar_bgra_sem_corte(sprite_esc, angulo_atual)
            h_anim, w_anim = sprite_anim.shape[:2]

            centro_x = int(round(p.ancora_x_norm * w_frame + p.offset_x_px + desloc_x))
            centro_y = int(round(p.ancora_y_norm * h_frame + p.offset_y_px + desloc_y))

            # Mantém o pop-up em área segura sem cobrir o painel superior nem o medidor lateral
            centro_x = max(w_anim // 2 + 12, min(w_frame - 135 - w_anim // 2, centro_x))
            centro_y = max(92 + h_anim // 2, min(h_frame - 48 - h_anim // 2, centro_y))

            x0 = centro_x - w_anim // 2
            y0 = centro_y - h_anim // 2
            _sobrepor_bgra_roi(frame, sprite_anim, x0, y0, alpha)

            if p.rotulo:
                _desenhar_tag_translucida(frame, p.rotulo, centro_x, y0 + h_anim + 4, cor_tag, alpha)

        self._particulas = restantes
