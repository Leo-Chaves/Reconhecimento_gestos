"""Motor de áudio em memória para o Theremin Gestual.

Inicializa o ``pygame.mixer`` com baixa latência e sintetiza em memória
(via NumPy, sem arquivos ``.wav`` externos) as ondas senoidais da escala
de Dó Maior, com suporte a modulação contínua de volume em tempo real.
"""

from __future__ import annotations

import numpy as np
import pygame

TAXA_AMOSTRAGEM: int = 44100
TAMANHO_AMOSTRA: int = -16
CANAIS_MIXER: int = 2
BUFFER_MIXER: int = 512
DURACAO_BUFFER_SEGUNDOS: float = 1.0
AMPLITUDE_MAXIMA: int = 14000
FADE_IN_MS: int = 25
FADE_OUT_MS: int = 60

NOTAS_FREQUENCIAS: dict[str, float] = {
    "DO": 261.63,
    "RE": 293.66,
    "MI": 329.63,
    "FA": 349.23,
    "SOL": 392.00,
    "LA": 440.00,
    "SI": 493.88,
}

_inicializado: bool = False
_sons: dict[str, pygame.mixer.Sound] = {}
_canais: list[pygame.mixer.Channel] = []
_canal_idx: int = 0
_nota_ativa: str | None = None
_som_ativo: pygame.mixer.Sound | None = None


def _limitar_volume(volume: float) -> float:
    """Garante que o volume permaneça na faixa operacional [0.1, 1.0]."""
    return max(0.1, min(1.0, float(volume)))


def _gerar_onda_senoidal(frequencia: float) -> pygame.mixer.Sound:
    """Gera uma onda senoidal estéreo com número inteiro de ciclos para loop sem cliques."""
    num_ciclos = max(1, round(frequencia * DURACAO_BUFFER_SEGUNDOS))
    num_amostras = max(1, round(num_ciclos * TAXA_AMOSTRAGEM / frequencia))
    fase = np.linspace(0.0, 2.0 * np.pi * num_ciclos, num_amostras, endpoint=False)
    forma_onda = (AMPLITUDE_MAXIMA * np.sin(fase)).astype(np.int16)
    estereo = np.ascontiguousarray(np.column_stack((forma_onda, forma_onda)))
    return pygame.sndarray.make_sound(estereo)


def inicializar_audio() -> None:
    """Inicializa o mixer de baixa latência e sintetiza todas as notas em memória."""
    global _inicializado, _sons, _canais

    if _inicializado and pygame.mixer.get_init() is not None:
        return

    pygame.mixer.pre_init(
        frequency=TAXA_AMOSTRAGEM,
        size=TAMANHO_AMOSTRA,
        channels=CANAIS_MIXER,
        buffer=BUFFER_MIXER,
    )
    pygame.mixer.init(
        frequency=TAXA_AMOSTRAGEM,
        size=TAMANHO_AMOSTRA,
        channels=CANAIS_MIXER,
        buffer=BUFFER_MIXER,
    )
    pygame.mixer.set_num_channels(8)

    _sons = {
        nota: _gerar_onda_senoidal(freq)
        for nota, freq in NOTAS_FREQUENCIAS.items()
    }
    _canais = [pygame.mixer.Channel(0), pygame.mixer.Channel(1)]
    _inicializado = True


def tocar_nota(nota: str, volume: float = 0.8) -> None:
    """Dispara a nota solicitada em sustentação contínua aplicando o volume inicial.

    Args:
        nota: Nome da nota na escala de Dó Maior (``DO`` a ``SI``).
        volume: Intensidade sonora inicial entre ``0.1`` e ``1.0``.
    """
    global _canal_idx, _nota_ativa, _som_ativo

    inicializar_audio()
    chave = nota.upper()
    if chave not in _sons:
        raise ValueError(f"Nota inválida '{nota}'. Use uma de: {list(NOTAS_FREQUENCIAS)}")

    vol = _limitar_volume(volume)
    som = _sons[chave]

    canal_anterior = _canais[_canal_idx]
    if canal_anterior.get_busy():
        canal_anterior.fadeout(FADE_OUT_MS)

    _canal_idx = 1 - _canal_idx
    canal_novo = _canais[_canal_idx]
    canal_novo.set_volume(1.0)
    som.set_volume(vol)
    canal_novo.play(som, loops=-1, fade_ms=FADE_IN_MS)

    _nota_ativa = chave
    _som_ativo = som


def atualizar_volume(volume: float) -> None:
    """Ajusta em tempo real o volume do som ativo enquanto a nota está soando."""
    if not _inicializado or _som_ativo is None:
        return

    vol = _limitar_volume(volume)
    _som_ativo.set_volume(vol)


def parar_nota() -> None:
    """Interrompe suavemente a nota em reprodução (silêncio)."""
    global _nota_ativa, _som_ativo

    if not _inicializado:
        return

    for canal in _canais:
        if canal.get_busy():
            canal.fadeout(FADE_OUT_MS)

    _nota_ativa = None
    _som_ativo = None


def encerrar_audio() -> None:
    """Libera os recursos do mixer ao finalizar a aplicação."""
    global _inicializado, _sons, _canais, _nota_ativa, _som_ativo

    if _inicializado and pygame.mixer.get_init() is not None:
        parar_nota()
        pygame.mixer.quit()

    _sons.clear()
    _canais.clear()
    _nota_ativa = None
    _som_ativo = None
    _inicializado = False


inicializar_audio()
