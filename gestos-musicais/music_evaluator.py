"""Avaliador musical em tempo real (Modo Livre) para o Theremin Gestual.

Analisa uma janela deslizante das últimas notas confirmadas na escala de Dó Maior
para classificar a frase musical como agradável (``good`` -> animal fofo) ou
dissonante/repetitiva (``bad`` -> gatinho triste).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Literal, Sequence

CategoriaMeme = Literal["good", "bad"]

GRAUS_ESCALA: dict[str, int] = {
    "DO": 0,
    "RE": 1,
    "MI": 2,
    "FA": 3,
    "SOL": 4,
    "LA": 5,
    "SI": 6,
}

TRIADES_CONSONANTES: tuple[frozenset[str], ...] = (
    frozenset({"DO", "MI", "SOL"}),
    frozenset({"RE", "FA", "LA"}),
    frozenset({"MI", "SOL", "SI"}),
    frozenset({"FA", "LA", "DO"}),
    frozenset({"SOL", "SI", "RE"}),
    frozenset({"LA", "DO", "MI"}),
)


@dataclass(frozen=True)
class ResultadoAvaliacao:
    """Resultado da análise de uma sequência de notas."""

    categoria: CategoriaMeme | None
    motivo: str
    pontuacao: int


def avaliar_sequencia(notas: Sequence[str]) -> ResultadoAvaliacao:
    """Avalia uma sequência de 4 a 6 notas da escala de Dó Maior.

    Critérios principais:
    - Repetição excessiva (<= 2 notas distintas ou oscilação A-B-A-B) -> ``bad``
    - Trítono repetido (FA <-> SI) ou múltiplos saltos disjuntos -> ``bad``
    - Escala contínua, arpejo de tríade consonante ou melodia variada -> ``good``
    """
    if len(notas) < 4:
        return ResultadoAvaliacao(None, "", 0)

    notas_validas = [n for n in notas if n in GRAUS_ESCALA]
    if len(notas_validas) < 4:
        return ResultadoAvaliacao(None, "", 0)

    unicas = set(notas_validas)
    graus = [GRAUS_ESCALA[n] for n in notas_validas]
    passos = [b - a for a, b in zip(graus, graus[1:])]
    abs_passos = [abs(p) for p in passos]

    tritonos = sum(
        1 for a, b in zip(notas_validas, notas_validas[1:])
        if {a, b} == {"FA", "SI"}
    )
    oscilacao_ab = (
        len(notas_validas) >= 4
        and notas_validas[-1] == notas_validas[-3]
        and notas_validas[-2] == notas_validas[-4]
    )

    if len(unicas) <= 2 or oscilacao_ab:
        return ResultadoAvaliacao(
            "bad",
            "Loop robotico! Variedade zero...",
            -2,
        )

    if tritonos >= 2:
        return ResultadoAvaliacao(
            "bad",
            "Tritono do caos! O gatinho chorou.",
            -3,
        )

    # Verifica progressão escalar (ex.: DO-RE-MI-FA ou SOL-FA-MI-RE)
    passos_escalares = sum(1 for p in abs_passos if p == 1)
    mesma_direcao = (
        all(p > 0 for p in passos)
        or all(p < 0 for p in passos)
    )
    if passos_escalares >= len(passos) - 1 and len(unicas) >= 3 and tritonos == 0:
        if mesma_direcao or passos_escalares == len(passos):
            return ResultadoAvaliacao(
                "good",
                "Escala lisinha! Puro talento!",
                3,
            )

    # Verifica arpejo de tríade consonante (ex.: DO-MI-SOL-DO)
    if any(unicas.issubset(triade) for triade in TRIADES_CONSONANTES) and len(unicas) >= 3:
        return ResultadoAvaliacao(
            "good",
            "Arpejo harmonioso! Digno de palco!",
            3,
        )

    # Saltos amplos (>= 5 graus) ou trítono combinado com salto brusco
    saltos_caoticos = sum(1 for d in abs_passos if d >= 5)
    if tritonos >= 1 and saltos_caoticos >= 1:
        return ResultadoAvaliacao(
            "bad",
            "Dissonancia pesada! Coitado do gatinho...",
            -2,
        )

    if saltos_caoticos >= 2:
        return ResultadoAvaliacao(
            "bad",
            "Saltos caoticos! Afinacao foi passear.",
            -2,
        )

    if len(unicas) >= 3 and tritonos == 0:
        return ResultadoAvaliacao(
            "good",
            "Melodia aprovada pelos bichinhos!",
            2,
        )

    return ResultadoAvaliacao(
        "bad",
        "Sequencia estranha... tenta de novo!",
        -1,
    )


class AvaliadorMusical:
    """Mantém o histórico recente de notas e dispara avaliações com cooldown."""

    def __init__(
        self,
        tamanho_janela: int = 5,
        min_notas_avaliacao: int = 4,
        cooldown_segundos: float = 3.5,
        timeout_frase_segundos: float = 2.8,
    ) -> None:
        self.tamanho_janela = tamanho_janela
        self.min_notas_avaliacao = min_notas_avaliacao
        self.cooldown_segundos = cooldown_segundos
        self.timeout_frase_segundos = timeout_frase_segundos

        self._historico: deque[str] = deque(maxlen=tamanho_janela)
        self._ultimo_registro: float = 0.0
        self._ultimo_meme: float = 0.0
        self._notas_desde_meme: int = 0

    @property
    def historico(self) -> tuple[str, ...]:
        """Retorna as notas atualmente armazenadas na trilha."""
        return tuple(self._historico)

    def limpar(self) -> None:
        """Limpa o histórico de notas da frase atual."""
        self._historico.clear()
        self._notas_desde_meme = 0

    def registrar_nota(
        self,
        nota: str | None,
        agora: float | None = None,
    ) -> ResultadoAvaliacao | None:
        """Registra uma nova nota confirmada e retorna avaliação quando aplicável."""
        if nota is None or nota not in GRAUS_ESCALA:
            return None

        instante = time.monotonic() if agora is None else agora
        if self._ultimo_registro and (instante - self._ultimo_registro) > self.timeout_frase_segundos:
            self.limpar()

        self._ultimo_registro = instante
        self._historico.append(nota)
        self._notas_desde_meme += 1

        if len(self._historico) < self.min_notas_avaliacao:
            return None
        if self._notas_desde_meme < self.min_notas_avaliacao:
            return None
        if self._ultimo_meme and (instante - self._ultimo_meme) < self.cooldown_segundos:
            return None

        resultado = avaliar_sequencia(tuple(self._historico))
        if resultado.categoria is not None:
            self._ultimo_meme = instante
            self._notas_desde_meme = 0
            return resultado
        return None
