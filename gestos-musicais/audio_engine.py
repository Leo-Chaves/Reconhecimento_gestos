"""Motor de áudio em memória para o Theremin Gestual.

Inicializa o ``pygame.mixer`` com baixa latência e sintetiza em memória
(via NumPy, sem arquivos ``.wav`` externos) as notas da escala de Dó Maior em
múltiplos timbres e oitavas, além de efeitos sonoros curtos para os memes.
"""

from __future__ import annotations

import numpy as np
import pygame

TAXA_AMOSTRAGEM: int = 44100
TAMANHO_AMOSTRA: int = -16
CANAIS_MIXER: int = 2
BUFFER_MIXER: int = 512
DURACAO_BUFFER_SEGUNDOS: float = 1.0
AMPLITUDE_MAXIMA: int = 13500
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

TIMBRES: tuple[str, ...] = ("theremin", "flauta", "8bit")
NOMES_TIMBRES: dict[str, str] = {
    "theremin": "Theremin",
    "flauta": "Flauta Synth",
    "8bit": "Retro 8-bit",
}
OITAVAS_VALIDAS: tuple[int, ...] = (-1, 0, 1)

_inicializado: bool = False
_sons: dict[tuple[str, int, str], pygame.mixer.Sound] = {}
_sfx_memes: dict[str, pygame.mixer.Sound] = {}
_canais: list[pygame.mixer.Channel] = []
_canal_sfx: pygame.mixer.Channel | None = None
_canal_idx: int = 0

_nota_ativa: str | None = None
_som_ativo: pygame.mixer.Sound | None = None
_volume_atual: float = 0.8
_timbre_atual: str = "theremin"
_oitava_atual: int = 0


def _limitar_volume(volume: float) -> float:
    """Garante que o volume permaneça na faixa operacional [0.1, 1.0]."""
    return max(0.1, min(1.0, float(volume)))


def _array_para_som(amostras: np.ndarray) -> pygame.mixer.Sound:
    """Converte vetor float [-1, 1] em ``pygame.mixer.Sound`` estéreo 16-bit."""
    forma_onda = (np.clip(amostras, -1.0, 1.0) * AMPLITUDE_MAXIMA).astype(np.int16)
    estereo = np.ascontiguousarray(np.column_stack((forma_onda, forma_onda)))
    return pygame.sndarray.make_sound(estereo)


def _gerar_onda_nota(frequencia: float, timbre: str) -> pygame.mixer.Sound:
    """Sintetiza um buffer contínuo em loop perfeito para o timbre e frequência dados."""
    num_ciclos = max(1, round(frequencia * DURACAO_BUFFER_SEGUNDOS))
    num_amostras = max(1, round(num_ciclos * TAXA_AMOSTRAGEM / frequencia))
    u = np.linspace(0.0, 1.0, num_amostras, endpoint=False)
    fase = 2.0 * np.pi * num_ciclos * u

    if timbre == "theremin":
        # Onda senoidal com vibrato suave (5 ciclos exatos por segundo -> loop sem clique)
        onda = np.sin(fase + 0.08 * np.sin(2.0 * np.pi * 5.0 * u))
    elif timbre == "flauta":
        onda = (
            0.72 * np.sin(fase)
            + 0.20 * np.sin(2.0 * fase)
            + 0.08 * np.sin(3.0 * fase)
        )
    elif timbre == "8bit":
        base = 0.68 * np.sin(fase) + 0.22 * np.sin(3.0 * fase) + 0.10 * np.sin(5.0 * fase)
        onda = np.tanh(1.7 * base) * 0.78
    else:
        onda = np.sin(fase)

    return _array_para_som(onda)


def _gerar_sfx_fanfarra() -> pygame.mixer.Sound:
    """Sintetiza mini fanfarra alegre para sequências boas."""
    notas_arp = (523.25, 659.25, 783.99, 1046.50)
    segmentos: list[np.ndarray] = []
    for idx, freq in enumerate(notas_arp):
        dur = 0.085 if idx < len(notas_arp) - 1 else 0.24
        n = int(TAXA_AMOSTRAGEM * dur)
        t = np.arange(n, dtype=np.float64) / TAXA_AMOSTRAGEM
        env = np.minimum(1.0, t / 0.01) * np.exp(-t * (8.0 if idx < 3 else 5.5))
        seg = (0.75 * np.sin(2.0 * np.pi * freq * t) + 0.25 * np.sin(4.0 * np.pi * freq * t)) * env
        segmentos.append(seg)
    return _array_para_som(np.concatenate(segmentos) * 0.7)


def _gerar_sfx_wah_wah() -> pygame.mixer.Sound:
    """Sintetiza efeito cômico descendente 'wah-wah-waaaah' para sequências ruins."""
    passos = ((311.13, 0.17), (293.66, 0.17), (277.18, 0.18), (246.94, 0.42))
    segmentos: list[np.ndarray] = []
    for idx, (freq, dur) in enumerate(passos):
        n = int(TAXA_AMOSTRAGEM * dur)
        t = np.arange(n, dtype=np.float64) / TAXA_AMOSTRAGEM
        if idx == len(passos) - 1:
            freq_inst = freq * (1.0 - 0.12 * (t / dur))
            fase = 2.0 * np.pi * np.cumsum(freq_inst) / TAXA_AMOSTRAGEM
            env = np.minimum(1.0, t / 0.015) * np.exp(-t * 2.8)
        else:
            fase = 2.0 * np.pi * freq * t
            env = np.minimum(1.0, t / 0.012) * np.exp(-t * 5.0)
        seg = (0.75 * np.sin(fase) + 0.25 * np.sin(2.0 * fase)) * env
        segmentos.append(seg)
    return _array_para_som(np.concatenate(segmentos) * 0.72)


def inicializar_audio() -> None:
    """Inicializa o mixer de baixa latência e sintetiza notas e efeitos em memória."""
    global _inicializado, _sons, _sfx_memes, _canais, _canal_sfx

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
        (timbre, oitava, nota): _gerar_onda_nota(freq * (2.0 ** oitava), timbre)
        for timbre in TIMBRES
        for oitava in OITAVAS_VALIDAS
        for nota, freq in NOTAS_FREQUENCIAS.items()
    }
    _sfx_memes = {
        "good": _gerar_sfx_fanfarra(),
        "bad": _gerar_sfx_wah_wah(),
    }
    _canais = [pygame.mixer.Channel(0), pygame.mixer.Channel(1)]
    _canal_sfx = pygame.mixer.Channel(2)
    _inicializado = True


def definir_timbre(timbre: str | int) -> str:
    """Altera o timbre ativo (por nome ou índice 1..3) com transição imediata."""
    global _timbre_atual

    if isinstance(timbre, int):
        idx = max(1, min(len(TIMBRES), timbre)) - 1
        novo = TIMBRES[idx]
    else:
        novo = timbre.lower()
        if novo not in TIMBRES:
            raise ValueError(f"Timbre inválido '{timbre}'. Opções: {TIMBRES}")

    if novo != _timbre_atual:
        _timbre_atual = novo
        if _nota_ativa is not None:
            tocar_nota(_nota_ativa, _volume_atual)
    return _timbre_atual


def definir_oitava(oitava: int) -> int:
    """Define a oitava ativa (``-1``, ``0`` ou ``1``), ajustando nota em curso se houver."""
    global _oitava_atual

    nova = max(-1, min(1, int(oitava)))
    if nova != _oitava_atual:
        _oitava_atual = nova
        if _nota_ativa is not None:
            tocar_nota(_nota_ativa, _volume_atual)
    return _oitava_atual


def obter_frequencia_efetiva(nota: str) -> float:
    """Retorna a frequência em Hz da nota considerando a oitava ativa."""
    return NOTAS_FREQUENCIAS[nota.upper()] * (2.0 ** _oitava_atual)


def obter_estado_audio() -> tuple[str, int]:
    """Retorna ``(timbre_atual, oitava_atual)``."""
    return _timbre_atual, _oitava_atual


def tocar_nota(nota: str, volume: float = 0.8) -> None:
    """Dispara a nota solicitada em sustentação contínua aplicando o volume inicial."""
    global _canal_idx, _nota_ativa, _som_ativo, _volume_atual

    inicializar_audio()
    chave = nota.upper()
    if chave not in NOTAS_FREQUENCIAS:
        raise ValueError(f"Nota inválida '{nota}'. Use uma de: {list(NOTAS_FREQUENCIAS)}")

    vol = _limitar_volume(volume)
    _volume_atual = vol
    som = _sons[(_timbre_atual, _oitava_atual, chave)]

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
    global _volume_atual

    vol = _limitar_volume(volume)
    _volume_atual = vol
    if not _inicializado or _som_ativo is None:
        return
    _som_ativo.set_volume(vol)


def tocar_sfx_meme(categoria: str) -> None:
    """Dispara o efeito sonoro curto associado ao meme ('good' ou 'bad')."""
    inicializar_audio()
    sfx = _sfx_memes.get(categoria)
    if sfx is None or _canal_sfx is None:
        return
    _canal_sfx.set_volume(0.55)
    _canal_sfx.play(sfx)


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
    global _inicializado, _sons, _sfx_memes, _canais, _canal_sfx, _nota_ativa, _som_ativo

    if _inicializado and pygame.mixer.get_init() is not None:
        parar_nota()
        if _canal_sfx is not None and _canal_sfx.get_busy():
            _canal_sfx.stop()
        pygame.mixer.quit()

    _sons.clear()
    _sfx_memes.clear()
    _canais.clear()
    _canal_sfx = None
    _nota_ativa = None
    _som_ativo = None
    _inicializado = False


inicializar_audio()
