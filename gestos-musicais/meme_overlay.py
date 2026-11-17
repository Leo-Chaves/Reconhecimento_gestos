"""Gerenciador de cards pop-up animados de memes para o Theremin Gestual."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time

import cv2
import numpy as np

EXTENSOES_IMAGEM = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class _MemeAtivo:
    categoria: str
    titulo: str
    legenda: str
    imagem: np.ndarray | None
    inicio: float
    duracao: float


def _quebrar_texto(texto: str, max_chars: int = 24) -> list[str]:
    """Divide uma legenda em linhas curtas para caber no card do meme."""
    palavras = texto.split()
    linhas: list[str] = []
    atual = ""
    for palavra in palavras:
        candidato = f"{atual} {palavra}".strip()
        if len(candidato) <= max_chars:
            atual = candidato
        else:
            if atual:
                linhas.append(atual)
            atual = palavra
    if atual:
        linhas.append(atual)
    return linhas[:3]


def _sobrepor_imagem(
    destino: np.ndarray,
    origem: np.ndarray,
    x: int,
    y: int,
    alpha_global: float,
) -> None:
    """Desenha ``origem`` (BGR ou BGRA) sobre ``destino`` respeitando bordas e alpha."""
    h_dst, w_dst = destino.shape[:2]
    h_src, w_src = origem.shape[:2]

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

    recorte_src = origem[sy1:sy2, sx1:sx2]
    recorte_dst = destino[y1:y2, x1:x2]

    if recorte_src.shape[2] == 4:
        alpha = (recorte_src[:, :, 3:4].astype(np.float32) / 255.0) * alpha_global
        bgr = recorte_src[:, :, :3].astype(np.float32)
    else:
        alpha = np.full((recorte_src.shape[0], recorte_src.shape[1], 1), alpha_global, dtype=np.float32)
        bgr = recorte_src.astype(np.float32)

    mistura = bgr * alpha + recorte_dst.astype(np.float32) * (1.0 - alpha)
    destino[y1:y2, x1:x2] = np.clip(mistura, 0, 255).astype(np.uint8)


class GerenciadorMemes:
    """Carrega imagens locais de ``assets/memes/`` e renderiza cards animados."""

    def __init__(self, diretorio_memes: Path | None = None, duracao_segundos: float = 3.2) -> None:
        base = diretorio_memes or (Path(__file__).resolve().parent / "assets" / "memes")
        self.duracao_segundos = duracao_segundos
        self._imagens: dict[str, list[np.ndarray]] = {
            "good": self._carregar_pasta(base / "good"),
            "bad": self._carregar_pasta(base / "bad"),
        }
        self._indices: dict[str, int] = {"good": 0, "bad": 0}
        self._ativo: _MemeAtivo | None = None

    @staticmethod
    def _carregar_pasta(pasta: Path) -> list[np.ndarray]:
        if not pasta.exists():
            return []
        imagens: list[np.ndarray] = []
        for arquivo in sorted(pasta.iterdir()):
            if arquivo.suffix.lower() not in EXTENSOES_IMAGEM:
                continue
            img = cv2.imread(str(arquivo), cv2.IMREAD_UNCHANGED)
            if img is not None:
                imagens.append(img)
        return imagens

    def contar_imagens(self) -> dict[str, int]:
        """Retorna a quantidade de imagens carregadas por categoria."""
        return {cat: len(imgs) for cat, imgs in self._imagens.items()}

    def ativar(self, categoria: str, legenda: str, agora: float | None = None) -> None:
        """Ativa um pop-up animado para a categoria informada (``good`` ou ``bad``)."""
        if categoria not in self._imagens:
            return

        lista = self._imagens[categoria]
        imagem: np.ndarray | None = None
        if lista:
            idx = self._indices[categoria] % len(lista)
            self._indices[categoria] += 1
            if len(lista) > 1 and random.random() < 0.35:
                idx = random.randrange(len(lista))
            imagem = lista[idx]

        titulo = "BICHINHO FELIZ!" if categoria == "good" else "GATINHO TRISTE..."
        instante = time.monotonic() if agora is None else agora
        self._ativo = _MemeAtivo(
            categoria=categoria,
            titulo=titulo,
            legenda=legenda,
            imagem=imagem,
            inicio=instante,
            duracao=self.duracao_segundos,
        )

    def renderizar(self, frame: np.ndarray, agora: float | None = None) -> None:
        """Desenha o card pop-up do meme ativo sobre o frame, caso ainda esteja vigente."""
        if self._ativo is None:
            return

        instante = time.monotonic() if agora is None else agora
        decorrido = instante - self._ativo.inicio
        if decorrido < 0.0 or decorrido >= self._ativo.duracao:
            self._ativo = None
            return

        # Pop-in inicial (0.22s) e fade-out final (0.45s)
        if decorrido < 0.22:
            t = decorrido / 0.22
            escala = 0.78 + 0.22 * (1.0 - (1.0 - t) ** 2)
            alpha = min(1.0, t * 1.2)
        elif decorrido > (self._ativo.duracao - 0.45):
            restante = max(0.0, self._ativo.duracao - decorrido)
            escala = 1.0
            alpha = restante / 0.45
        else:
            escala = 1.0
            alpha = 1.0

        card_w = int(round(206 * escala))
        img_lado = int(round(166 * escala))
        linhas = _quebrar_texto(self._ativo.legenda, max_chars=23)
        card_h = int(round((224 + len(linhas) * 18) * escala))

        h_frame, w_frame = frame.shape[:2]
        x1 = 16
        y1 = 98
        x2 = min(w_frame - 90, x1 + card_w)
        y2 = min(h_frame - 40, y1 + card_h)
        if x2 - x1 < 80 or y2 - y1 < 80:
            return

        cor_borda = (75, 225, 130) if self._ativo.categoria == "good" else (85, 85, 245)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 22, 32), -1)
        cv2.addWeighted(overlay, 0.82 * alpha, frame, 1.0 - 0.82 * alpha, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor_borda, 2)

        cv2.putText(
            frame,
            self._ativo.titulo,
            (x1 + 12, y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * escala,
            cor_borda,
            2,
            cv2.LINE_AA,
        )

        img_x = x1 + ((x2 - x1) - img_lado) // 2
        img_y = y1 + 34
        if self._ativo.imagem is not None and img_lado > 10:
            img_redim = cv2.resize(self._ativo.imagem, (img_lado, img_lado), interpolation=cv2.INTER_AREA)
            _sobrepor_imagem(frame, img_redim, img_x, img_y, alpha)
        else:
            simbolo = ":D" if self._ativo.categoria == "good" else ":("
            cv2.putText(
                frame,
                simbolo,
                (x1 + 65, y1 + 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8 * escala,
                cor_borda,
                3,
                cv2.LINE_AA,
            )

        texto_y = img_y + img_lado + 20
        for linha in linhas:
            if texto_y < y2 - 6:
                cv2.putText(
                    frame,
                    linha,
                    (x1 + 12, texto_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44 * escala,
                    (235, 240, 248),
                    1,
                    cv2.LINE_AA,
                )
            texto_y += int(round(18 * escala))
