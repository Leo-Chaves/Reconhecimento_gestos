"""Módulo de rastreamento de mãos, histerese de gestos e controle expressivo.

Interpreta dinamicamente os landmarks do MediaPipe Hands para alternar entre:
- Modo 1 Mão (Simplificado): mapeia apenas as 8 notas (DO4 a DO5) em Modo
  Maior padrão, com volume fixo.
- Modo 2 Mãos (Expressivo):
  * Mão Esquerda: seleciona a nota musical (DO4 a DO5) ou Silêncio.
  * Mão Direita: controla o volume pela altura (eixo Y) e alterna a escala
    pela postura (Palma aberta = MAIOR; Punho fechado ou Pinça = MENOR).
  * Proximidade horizontal entre as mãos: ativa o Arpejador automático com
    velocidade progressiva conforme as mãos se aproximam.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Sequence

import numpy as np

PapelMao = Literal["unica", "esquerda", "direita"]
TipoEscala = Literal["maior", "menor"]
PosturaDireita = Literal["aberta", "fechada"]
EstadoDedos = tuple[bool, bool, bool, bool, bool]

# ---------------------------------------------------------------------------
# Parâmetros de Calibração: Histerese dos Dedos (escore anatômico individual)
# ---------------------------------------------------------------------------
# Combina extensão normalizada pelo segmento MCP->PIP do próprio dedo com o
# alinhamento angular MCP->PIP->TIP, tornando mindinho e indicador equivalentes.
LIMIAR_ABRIR_DEDO: float = 0.55
LIMIAR_FECHAR_DEDO: float = 0.35
LIMIAR_MEDIO_DEDO: float = 0.45

# Limiares do escore composto multi-referência do polegar (separa FA4 de SOL4).
LIMIAR_ABRIR_POLEGAR: float = 0.12
LIMIAR_FECHAR_POLEGAR: float = 0.00
LIMIAR_MEDIO_POLEGAR: float = 0.06

FRAMES_CONFIRMACAO_ESCALA_DIR: int = 3
# Frames tolerados caso o MediaPipe perca a mão por movimento rápido (anti-piscada).
FRAMES_MEMORIA_PERDA_MAO: int = 2

# ---------------------------------------------------------------------------
# Parâmetros de Mapeamento de Volume, Suavização e Arpejador Espacial
# ---------------------------------------------------------------------------
VOLUME_PADRAO_1_MAO: float = 0.75
Y_TOPO: float = 0.15
Y_BASE: float = 0.85
VOL_MIN: float = 0.10
VOL_MAX: float = 1.00
EMA_VOLUME_ANTERIOR: float = 0.70
EMA_VOLUME_NOVO: float = 0.30
EMA_LANDMARKS_NOVO: float = 0.58

# Distância horizontal normalizada entre as palmas (eixo X) que ativa/desativa o Arpejador.
DIST_ATIVAR_ARPEJO: float = 0.35
DIST_DESATIVAR_ARPEJO: float = 0.39
DIST_ARPEJO_MAX_VEL: float = 0.15
TAXA_ARPEJO_MIN_HZ: float = 4.5
TAXA_ARPEJO_MAX_HZ: float = 9.5

# Índices dos 21 landmarks do MediaPipe Hands
PUNHO = 0
POLEGAR_MCP = 2
POLEGAR_IP = 3
POLEGAR_TIP = 4
INDICADOR_MCP = 5
INDICADOR_PIP = 6
INDICADOR_TIP = 8
MEDIO_MCP = 9
MEDIO_PIP = 10
MEDIO_TIP = 12
ANELAR_MCP = 13
ANELAR_PIP = 14
ANELAR_TIP = 16
MINDINHO_MCP = 17
MINDINHO_PIP = 18
MINDINHO_TIP = 20

TRIOS_DEDOS: tuple[tuple[int, int, int], ...] = (
    (INDICADOR_MCP, INDICADOR_PIP, INDICADOR_TIP),
    (MEDIO_MCP, MEDIO_PIP, MEDIO_TIP),
    (ANELAR_MCP, ANELAR_PIP, ANELAR_TIP),
    (MINDINHO_MCP, MINDINHO_PIP, MINDINHO_TIP),
)


@dataclass
class MaoRastreada:
    """Representação processada de uma mão para áudio e renderização."""

    papel: PapelMao
    pontos_norm: np.ndarray
    ancora_norm: tuple[float, float]
    nota: str | None = None
    escala: TipoEscala = "maior"
    postura_direita: PosturaDireita = "aberta"
    volume: float = VOLUME_PADRAO_1_MAO


@dataclass
class EstadoMaos:
    """Snapshot completo do estado gestual extraído em um frame."""

    modo_duas_maos: bool
    num_maos: int
    nota_candidata: str | None
    gesto_valido: bool
    escala: TipoEscala
    volume: float
    arpejo_ativo: bool
    taxa_arpejo_hz: float
    postura_direita: PosturaDireita
    ancora_meme: tuple[float, float] | None
    ponto_palma_esq: tuple[float, float] | None
    ponto_palma_dir: tuple[float, float] | None
    pontas_indicadores: list[tuple[float, float]]
    maos_visuais: list[MaoRastreada]


def landmarks_para_array(landmarks: Sequence[Any]) -> np.ndarray:
    """Converte uma sequência de 21 landmarks em array NumPy ``(21, 2)``."""
    pontos = np.empty((21, 2), dtype=np.float32)
    for idx in range(21):
        lm = landmarks[idx]
        if hasattr(lm, "x"):
            pontos[idx, 0] = float(lm.x)
            pontos[idx, 1] = float(lm.y)
        else:
            pontos[idx, 0] = float(lm[0])
            pontos[idx, 1] = float(lm[1])
    return pontos


def _dist_indices(pontos: np.ndarray, idx_a: int, idx_b: int) -> float:
    """Calcula a distância euclidiana 2D normalizada entre dois índices."""
    dx = float(pontos[idx_a, 0] - pontos[idx_b, 0])
    dy = float(pontos[idx_a, 1] - pontos[idx_b, 1])
    return math.hypot(dx, dy)


def _aplicar_histerese(valor: float, estava_aberto: bool | None, abrir: float, fechar: float, medio: float) -> bool:
    """Aplica zona morta entre ``fechar`` e ``abrir`` preservando o estado anterior."""
    if estava_aberto is None:
        return valor > medio
    if estava_aberto:
        return valor >= fechar
    return valor > abrir


def calcular_score_polegar(pontos: np.ndarray, tam_palma: float | None = None) -> float:
    """Calcula escore composto invariante à rotação para abertura do polegar.

    Combina três referências para evitar conflito entre FA4 e SOL4:
    1) alcance além da largura da palma ``dist(5, 17)``;
    2) avanço da ponta além da junta IP em relação ao mindinho e ao centro da palma;
    3) alinhamento angular da falange distal ``2 -> 3 -> 4``.
    """
    tam = max(1e-4, _dist_indices(pontos, PUNHO, MEDIO_MCP) if tam_palma is None else tam_palma)
    larg_palma = max(1e-4, _dist_indices(pontos, INDICADOR_MCP, MINDINHO_MCP))

    dist_tip_17 = _dist_indices(pontos, POLEGAR_TIP, MINDINHO_MCP)
    dist_ip_17 = _dist_indices(pontos, POLEGAR_IP, MINDINHO_MCP)
    dist_tip_9 = _dist_indices(pontos, POLEGAR_TIP, MEDIO_MCP)
    dist_ip_9 = _dist_indices(pontos, POLEGAR_IP, MEDIO_MCP)

    termo_largura = (dist_tip_17 - 1.15 * larg_palma) / larg_palma
    termo_junta = ((dist_tip_17 - dist_ip_17) + 0.60 * (dist_tip_9 - dist_ip_9)) / tam

    vax = float(pontos[POLEGAR_IP, 0] - pontos[POLEGAR_MCP, 0])
    vay = float(pontos[POLEGAR_IP, 1] - pontos[POLEGAR_MCP, 1])
    vbx = float(pontos[POLEGAR_TIP, 0] - pontos[POLEGAR_IP, 0])
    vby = float(pontos[POLEGAR_TIP, 1] - pontos[POLEGAR_IP, 1])
    norma = math.hypot(vax, vay) * math.hypot(vbx, vby)
    cos_falange = (vax * vbx + vay * vby) / norma if norma > 1e-6 else 0.0
    termo_angulo = 0.25 * (cos_falange - 0.65)

    return 0.45 * termo_largura + 0.45 * termo_junta + termo_angulo


def calcular_score_dedo(pontos: np.ndarray, mcp: int, pip: int, tip: int) -> float:
    """Calcula escore anatômico individual para extensão de um dedo principal.

    Normaliza o ganho radial pelo comprimento proximal ``dist(mcp, pip)`` do
    próprio dedo (evitando prejudicar dedos curtos como o mindinho em SI4/LA4)
    e soma o alinhamento angular na junta ``pip`` para rejeitar dedos curvados.
    """
    comp_proximal = max(1e-4, _dist_indices(pontos, mcp, pip))
    ext_relativa = (_dist_indices(pontos, PUNHO, tip) - _dist_indices(pontos, PUNHO, pip)) / comp_proximal

    v1x = float(pontos[pip, 0] - pontos[mcp, 0])
    v1y = float(pontos[pip, 1] - pontos[mcp, 1])
    v2x = float(pontos[tip, 0] - pontos[pip, 0])
    v2y = float(pontos[tip, 1] - pontos[pip, 1])
    norma = math.hypot(v1x, v1y) * math.hypot(v2x, v2y)
    cos_pip = (v1x * v2x + v1y * v2y) / norma if norma > 1e-6 else 0.0

    return 0.55 * ext_relativa + 0.45 * cos_pip


def detectar_dedos_com_histerese(
    landmarks: Sequence[Any] | np.ndarray,
    anterior: EstadoDedos | None = None,
) -> EstadoDedos:
    """Detecta ``(polegar, indicador, medio, anelar, mindinho)`` com histerese."""
    pontos = landmarks if isinstance(landmarks, np.ndarray) else landmarks_para_array(landmarks)
    tam_palma = max(1e-4, _dist_indices(pontos, PUNHO, MEDIO_MCP))

    score_pol = calcular_score_polegar(pontos, tam_palma)
    pol = _aplicar_histerese(
        score_pol,
        None if anterior is None else anterior[0],
        LIMIAR_ABRIR_POLEGAR,
        LIMIAR_FECHAR_POLEGAR,
        LIMIAR_MEDIO_POLEGAR,
    )

    dedos_principais: list[bool] = []
    for idx, (mcp, pip, tip) in enumerate(TRIOS_DEDOS):
        score_dedo = calcular_score_dedo(pontos, mcp, pip, tip)
        estado_ant = None if anterior is None else anterior[idx + 1]
        dedos_principais.append(
            _aplicar_histerese(score_dedo, estado_ant, LIMIAR_ABRIR_DEDO, LIMIAR_FECHAR_DEDO, LIMIAR_MEDIO_DEDO)
        )

    return (pol, dedos_principais[0], dedos_principais[1], dedos_principais[2], dedos_principais[3])


def classificar_dedos_para_nota(dedos: EstadoDedos) -> tuple[bool, str | None]:
    """Mapeia os 5 estados booleanos dos dedos para ``DO4``..``DO5`` ou Silêncio."""
    pol, ind, med, ane, min_ = dedos
    principais = (ind, med, ane, min_)

    if principais == (False, False, False, False):
        return True, None
    if principais == (False, False, False, True) and pol:
        return True, "LA4"
    if principais == (True, False, False, True):
        return True, "SI4"
    if principais == (True, False, False, False):
        return True, ("DO5" if pol else "DO4")
    if principais == (True, True, False, False):
        return True, "RE4"
    if principais == (True, True, True, False):
        return True, "MI4"
    if principais == (True, True, True, True):
        return True, ("SOL4" if pol else "FA4")

    return False, None


def classificar_gesto_nota(
    landmarks: Sequence[Any] | np.ndarray,
    estado_dedos_anterior: EstadoDedos | None = None,
) -> tuple[bool, str | None]:
    """Classifica os 21 landmarks em ``DO4``..``DO5`` ou Silêncio com histerese."""
    dedos = detectar_dedos_com_histerese(landmarks, estado_dedos_anterior)
    return classificar_dedos_para_nota(dedos)


def classificar_escala_mao_direita(
    landmarks: Sequence[Any] | np.ndarray,
    escala_anterior: TipoEscala = "maior",
    estado_dedos_anterior: EstadoDedos | None = None,
) -> tuple[TipoEscala, EstadoDedos]:
    """Define ``'maior'`` (palma aberta) ou ``'menor'`` (punho/pinça) pela mão direita."""
    pontos = landmarks if isinstance(landmarks, np.ndarray) else landmarks_para_array(landmarks)
    dedos = detectar_dedos_com_histerese(pontos, estado_dedos_anterior)

    tam_palma = max(1e-4, _dist_indices(pontos, PUNHO, MEDIO_MCP))
    dist_pinca = _dist_indices(pontos, POLEGAR_TIP, INDICADOR_TIP)
    em_pinca = dist_pinca < max(0.045, 0.30 * tam_palma)
    total_abertos = sum(1 for aberto in dedos if aberto)

    if em_pinca or total_abertos <= 1:
        return "menor", dedos
    if total_abertos >= 3:
        return "maior", dedos
    return escala_anterior, dedos


def calcular_volume_por_altura(y_punho: float, volume_anterior: float) -> float:
    """Mapeia ``y`` do punho para ``[0.10, 1.00]`` com suavização EMA."""
    y_limitado = max(Y_TOPO, min(Y_BASE, float(y_punho)))
    fator = (Y_BASE - y_limitado) / (Y_BASE - Y_TOPO)
    vol_bruto = VOL_MIN + fator * (VOL_MAX - VOL_MIN)
    vol_suave = EMA_VOLUME_ANTERIOR * volume_anterior + EMA_VOLUME_NOVO * vol_bruto
    return max(VOL_MIN, min(VOL_MAX, vol_suave))


def _obter_rotulo_handedness(handedness_list: Sequence[Any] | None, indice: int) -> str:
    """Lê o rótulo textual ('Left' ou 'Right') com segurança."""
    if not handedness_list or indice >= len(handedness_list):
        return ""
    item = handedness_list[indice]
    if isinstance(item, str):
        return item
    classificacoes = getattr(item, "classification", None)
    if classificacoes:
        return str(classificacoes[0].label)
    return ""


def separar_duas_maos(
    maos_landmarks: Sequence[Any],
    handedness_list: Sequence[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna ``(pontos_mao_esquerda, pontos_mao_direita)`` para frame já espelhado."""
    pts_0 = landmarks_para_array(
        maos_landmarks[0].landmark if hasattr(maos_landmarks[0], "landmark") else maos_landmarks[0]
    )
    pts_1 = landmarks_para_array(
        maos_landmarks[1].landmark if hasattr(maos_landmarks[1], "landmark") else maos_landmarks[1]
    )

    rot_0 = _obter_rotulo_handedness(handedness_list, 0)
    rot_1 = _obter_rotulo_handedness(handedness_list, 1)

    if {rot_0, rot_1} == {"Left", "Right"}:
        esq, dir_ = (pts_0, pts_1) if rot_0 == "Left" else (pts_1, pts_0)
        if float(esq[PUNHO, 0]) <= float(dir_[PUNHO, 0]) + 0.18:
            return esq, dir_

    if float(pts_0[PUNHO, 0]) <= float(pts_1[PUNHO, 0]):
        return pts_0, pts_1
    return pts_1, pts_0


class RastreadorMaos:
    """Gerencia adaptação dinâmica 1 vs. 2 mãos, histerese, arpejador e EMA."""

    def __init__(self) -> None:
        self._escala_atual: TipoEscala = "maior"
        self._escala_candidata: TipoEscala = "maior"
        self._contador_escala: int = 0
        self._volume_suave: float = 0.65
        self._arpejo_ativo: bool = False
        self._pontos_suaves: dict[PapelMao, np.ndarray] = {}
        self._dedos_esq_anterior: EstadoDedos | None = None
        self._dedos_dir_anterior: EstadoDedos | None = None
        self._dedos_unica_anterior: EstadoDedos | None = None
        self._ultimo_estado_valido: EstadoMaos | None = None
        self._frames_sem_mao: int = 0

    def _suavizar_pontos(self, papel: PapelMao, pontos_brutos: np.ndarray) -> np.ndarray:
        anterior = self._pontos_suaves.get(papel)
        if anterior is None:
            suavizado = pontos_brutos.copy()
        else:
            suavizado = (1.0 - EMA_LANDMARKS_NOVO) * anterior + EMA_LANDMARKS_NOVO * pontos_brutos
        self._pontos_suaves[papel] = suavizado
        return suavizado

    def processar(
        self,
        maos_landmarks: Sequence[Any],
        handedness_list: Sequence[Any] | None = None,
        agora: float | None = None,
    ) -> EstadoMaos:
        """Processa os landmarks do frame atual e retorna o estado consolidado."""
        del agora
        num_maos = len(maos_landmarks)

        if num_maos == 0:
            self._frames_sem_mao += 1
            if (
                self._ultimo_estado_valido is not None
                and self._frames_sem_mao <= FRAMES_MEMORIA_PERDA_MAO
            ):
                return self._ultimo_estado_valido

            self._ultimo_estado_valido = None
            self._arpejo_ativo = False
            self._pontos_suaves.clear()
            self._dedos_esq_anterior = None
            self._dedos_dir_anterior = None
            self._dedos_unica_anterior = None
            self._escala_atual = "maior"
            self._escala_candidata = "maior"
            self._contador_escala = 0
            return EstadoMaos(
                modo_duas_maos=False,
                num_maos=0,
                nota_candidata=None,
                gesto_valido=True,
                escala="maior",
                volume=VOLUME_PADRAO_1_MAO,
                arpejo_ativo=False,
                taxa_arpejo_hz=0.0,
                postura_direita="aberta",
                ancora_meme=None,
                ponto_palma_esq=None,
                ponto_palma_dir=None,
                pontas_indicadores=[],
                maos_visuais=[],
            )

        self._frames_sem_mao = 0

        # Modo simplificado com 1 mão: toca DO4..DO5 no Modo Maior padrão
        if num_maos == 1:
            bruto = maos_landmarks[0].landmark if hasattr(maos_landmarks[0], "landmark") else maos_landmarks[0]
            pts_brutos = landmarks_para_array(bruto)
            pts_visuais = self._suavizar_pontos("unica", pts_brutos)
            self._pontos_suaves.pop("esquerda", None)
            self._pontos_suaves.pop("direita", None)
            self._dedos_esq_anterior = None
            self._dedos_dir_anterior = None

            dedos_unica = detectar_dedos_com_histerese(pts_brutos, self._dedos_unica_anterior)
            self._dedos_unica_anterior = dedos_unica
            valido, nota = classificar_dedos_para_nota(dedos_unica)

            self._escala_atual = "maior"
            self._escala_candidata = "maior"
            self._contador_escala = 0
            self._arpejo_ativo = False

            ancora = (float(pts_visuais[MEDIO_MCP, 0]), float(pts_visuais[MEDIO_MCP, 1]))
            ponta_ind = (float(pts_visuais[INDICADOR_TIP, 0]), float(pts_visuais[INDICADOR_TIP, 1]))
            mao_visual = MaoRastreada(
                papel="unica",
                pontos_norm=pts_visuais,
                ancora_norm=ancora,
                nota=nota,
                escala="maior",
                postura_direita="aberta",
                volume=VOLUME_PADRAO_1_MAO,
            )
            estado = EstadoMaos(
                modo_duas_maos=False,
                num_maos=1,
                nota_candidata=nota,
                gesto_valido=valido,
                escala="maior",
                volume=VOLUME_PADRAO_1_MAO,
                arpejo_ativo=False,
                taxa_arpejo_hz=0.0,
                postura_direita="aberta",
                ancora_meme=ancora,
                ponto_palma_esq=ancora,
                ponto_palma_dir=None,
                pontas_indicadores=[ponta_ind],
                maos_visuais=[mao_visual],
            )
            self._ultimo_estado_valido = estado
            return estado

        # Modo expressivo com 2 mãos:
        # - Mão Esquerda: nota (DO4..DO5 ou Silêncio)
        # - Mão Direita: volume (Y) + postura (Aberta=MAIOR / Fechada ou Pinça=MENOR)
        # - Aproximação horizontal (eixo X): liga o Arpejador com velocidade progressiva
        esq_bruta, dir_bruta = separar_duas_maos(maos_landmarks[:2], handedness_list)
        pts_esq = self._suavizar_pontos("esquerda", esq_bruta)
        pts_dir = self._suavizar_pontos("direita", dir_bruta)
        self._pontos_suaves.pop("unica", None)
        self._dedos_unica_anterior = None

        dedos_esq = detectar_dedos_com_histerese(esq_bruta, self._dedos_esq_anterior)
        self._dedos_esq_anterior = dedos_esq
        valido, nota = classificar_dedos_para_nota(dedos_esq)

        escala_cand, dedos_dir = classificar_escala_mao_direita(
            dir_bruta,
            self._escala_atual,
            self._dedos_dir_anterior,
        )
        self._dedos_dir_anterior = dedos_dir

        if escala_cand == self._escala_candidata:
            self._contador_escala += 1
        else:
            self._escala_candidata = escala_cand
            self._contador_escala = 1
        if self._contador_escala >= FRAMES_CONFIRMACAO_ESCALA_DIR:
            self._escala_atual = self._escala_candidata

        self._volume_suave = calcular_volume_por_altura(float(dir_bruta[PUNHO, 1]), self._volume_suave)

        # Arpejador por proximidade horizontal entre palmas com histerese
        dist_x = abs(float(dir_bruta[MEDIO_MCP, 0] - esq_bruta[MEDIO_MCP, 0]))
        if not self._arpejo_ativo and dist_x <= DIST_ATIVAR_ARPEJO:
            self._arpejo_ativo = True
        elif self._arpejo_ativo and dist_x >= DIST_DESATIVAR_ARPEJO:
            self._arpejo_ativo = False

        if self._arpejo_ativo:
            fator_vel = max(
                0.0,
                min(
                    1.0,
                    (DIST_DESATIVAR_ARPEJO - dist_x)
                    / max(1e-4, DIST_DESATIVAR_ARPEJO - DIST_ARPEJO_MAX_VEL),
                ),
            )
            taxa_arpejo = TAXA_ARPEJO_MIN_HZ + fator_vel * (TAXA_ARPEJO_MAX_HZ - TAXA_ARPEJO_MIN_HZ)
        else:
            taxa_arpejo = 0.0

        postura_dir: PosturaDireita = "aberta" if self._escala_atual == "maior" else "fechada"
        ancora_esq = (float(pts_esq[MEDIO_MCP, 0]), float(pts_esq[MEDIO_MCP, 1]))
        ancora_dir = (float(pts_dir[MEDIO_MCP, 0]), float(pts_dir[MEDIO_MCP, 1]))
        pontas_ind = [
            (float(pts_esq[INDICADOR_TIP, 0]), float(pts_esq[INDICADOR_TIP, 1])),
            (float(pts_dir[INDICADOR_TIP, 0]), float(pts_dir[INDICADOR_TIP, 1])),
        ]

        mao_esq_visual = MaoRastreada(
            papel="esquerda",
            pontos_norm=pts_esq,
            ancora_norm=ancora_esq,
            nota=nota,
            escala=self._escala_atual,
            postura_direita=postura_dir,
            volume=self._volume_suave,
        )
        mao_dir_visual = MaoRastreada(
            papel="direita",
            pontos_norm=pts_dir,
            ancora_norm=ancora_dir,
            nota=None,
            escala=self._escala_atual,
            postura_direita=postura_dir,
            volume=self._volume_suave,
        )

        estado = EstadoMaos(
            modo_duas_maos=True,
            num_maos=num_maos,
            nota_candidata=nota,
            gesto_valido=valido,
            escala=self._escala_atual,
            volume=self._volume_suave,
            arpejo_ativo=self._arpejo_ativo,
            taxa_arpejo_hz=taxa_arpejo,
            postura_direita=postura_dir,
            ancora_meme=ancora_esq,
            ponto_palma_esq=ancora_esq,
            ponto_palma_dir=ancora_dir,
            pontas_indicadores=pontas_ind,
            maos_visuais=[mao_esq_visual, mao_dir_visual],
        )
        self._ultimo_estado_valido = estado
        return estado
