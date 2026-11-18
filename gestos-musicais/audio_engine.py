"""Motor de áudio polifônico em memória com 5 instrumentos, reverb e arpejador.

Sintetiza em memória via NumPy (sem arquivos externos):
- 5 instrumentos distintos: Piano, Violão (Karplus-Strong), Cordas, Synth e 8-Bit.
- Reverb estéreo espacial (circular para loops contínuos e com cauda natural
  para notas dedilhadas/percussivas).
- Arquitetura de 16 canais no ``pygame.mixer`` separando Ao Vivo (sustentado +
  pool polifônico de arpejos) e Mini-Looper (sustentado + pool de arpejos).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pygame

ModoHarmonico = Literal["acorde", "nota"]
TipoEscala = Literal["maior", "menor"]

TAXA_AMOSTRAGEM: int = 44100
TAMANHO_AMOSTRA: int = -16
CANAIS_MIXER: int = 2
BUFFER_MIXER: int = 512
DURACAO_BUFFER_LOOP_SEG: float = 1.0
DURACAO_PLUCKED_SEG: float = 0.68
AMPLITUDE_MAXIMA: int = 12000
FADE_IN_MS: int = 22
FADE_OUT_MS: int = 55

# Escala completa de Dó4 a Dó5 (8 notas)
NOTAS_FREQUENCIAS: dict[str, float] = {
    "DO4": 261.63,
    "RE4": 293.66,
    "MI4": 329.63,
    "FA4": 349.23,
    "SOL4": 392.00,
    "LA4": 440.00,
    "SI4": 493.88,
    "DO5": 523.25,
}

NOTAS_MENOR_NATURAL: dict[str, float] = {
    "DO4": 261.63,
    "RE4": 293.66,
    "MI4": 311.13,
    "FA4": 349.23,
    "SOL4": 392.00,
    "LA4": 415.30,
    "SI4": 466.16,
    "DO5": 523.25,
}

NOTAS_MIDI: dict[str, int] = {
    "DO4": 60,
    "RE4": 62,
    "MI4": 64,
    "FA4": 65,
    "SOL4": 67,
    "LA4": 69,
    "SI4": 71,
    "DO5": 72,
}

NOTAS_MIDI_MENOR_NATURAL: dict[str, int] = {
    "DO4": 60,
    "RE4": 62,
    "MI4": 63,
    "FA4": 65,
    "SOL4": 67,
    "LA4": 68,
    "SI4": 70,
    "DO5": 72,
}

TIMBRES: tuple[str, ...] = ("piano", "violao", "cordas", "synth", "8bit")
NOMES_TIMBRES: dict[str, str] = {
    "piano": "Piano",
    "violao": "Violao",
    "cordas": "Cordas",
    "synth": "Synth",
    "8bit": "8-Bit",
}
MODOS_HARMONICOS: tuple[ModoHarmonico, ...] = ("acorde", "nota")
ESCALAS_VALIDAS: tuple[TipoEscala, ...] = ("maior", "menor")

_PESOS_TRIADE: tuple[float, float, float] = (0.46, 0.29, 0.25)
_TAPS_REVERB_ESQ: tuple[tuple[int, float], ...] = ((1367, 0.22), (2903, 0.14), (4721, 0.08))
_TAPS_REVERB_DIR: tuple[tuple[int, float], ...] = ((1693, 0.22), (3371, 0.14), (5237, 0.08))

_inicializado: bool = False
_sons_loop: dict[tuple[str, ModoHarmonico, TipoEscala, str], pygame.mixer.Sound] = {}
_sons_plucked: dict[tuple[str, int], pygame.mixer.Sound] = {}

_canais_loop_vivo: tuple[pygame.mixer.Channel, pygame.mixer.Channel] | None = None
_canais_loop_looper: tuple[pygame.mixer.Channel, pygame.mixer.Channel] | None = None
_canais_plucked_vivo: list[pygame.mixer.Channel] = []
_canais_plucked_looper: list[pygame.mixer.Channel] = []

_idx_loop_vivo: int = 0
_idx_loop_looper: int = 0
_idx_plucked_vivo: int = 0
_idx_plucked_looper: int = 0

_nota_ativa_vivo: str | None = None
_escala_ativa_vivo: TipoEscala = "maior"
_arpejo_ativo_vivo: bool = False
_volume_atual_vivo: float = 0.75
_timbre_atual: str = "piano"
_modo_harmonico_atual: ModoHarmonico = "acorde"


def _limitar_volume(volume: float) -> float:
    """Garante que o volume permaneça na faixa operacional [0.08, 1.0]."""
    return max(0.08, min(1.0, float(volume)))


def _midi_para_freq(midi: int) -> float:
    """Converte número de nota MIDI para frequência em Hz (A4 = 69 -> 440 Hz)."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _aplicar_reverb_estereo(onda: np.ndarray, circular: bool) -> np.ndarray:
    """Aplica eco/reverb espacial estéreo preservando fase em loops ou com cauda limpa."""
    esq = onda.copy()
    dir_ = onda.copy()
    n = len(onda)

    if circular:
        for atraso, ganho in _TAPS_REVERB_ESQ:
            esq += ganho * np.roll(onda, atraso)
        for atraso, ganho in _TAPS_REVERB_DIR:
            dir_ += ganho * np.roll(onda, atraso)
    else:
        for atraso, ganho in _TAPS_REVERB_ESQ:
            if atraso < n:
                esq[atraso:] += ganho * onda[:-atraso]
        for atraso, ganho in _TAPS_REVERB_DIR:
            if atraso < n:
                dir_[atraso:] += ganho * onda[:-atraso]

    estereo = np.column_stack((esq, dir_)) * (1.0 / 1.44)
    return np.clip(estereo, -1.0, 1.0)


def _estereo_para_som(estereo: np.ndarray) -> pygame.mixer.Sound:
    """Converte matriz float ``(N, 2)`` em ``pygame.mixer.Sound`` 16-bit."""
    pcm = np.ascontiguousarray((np.clip(estereo, -1.0, 1.0) * AMPLITUDE_MAXIMA).astype(np.int16))
    som = pygame.sndarray.make_sound(pcm)
    som.set_volume(1.0)
    return som


def _frequencias_para_modo(
    nota: str,
    escala: TipoEscala,
    modo_harmonico: ModoHarmonico,
) -> list[tuple[float, float]]:
    """Retorna lista de ``(frequencia_hz, peso)`` para acorde ou nota pura."""
    chave = nota.upper()
    if modo_harmonico == "nota":
        tabela = NOTAS_FREQUENCIAS if escala == "maior" else NOTAS_MENOR_NATURAL
        return [(tabela[chave], 1.0)]

    freq_base = NOTAS_FREQUENCIAS[chave]
    semitons_terca = 4 if escala == "maior" else 3
    freqs = (
        freq_base,
        freq_base * (2.0 ** (semitons_terca / 12.0)),
        freq_base * (2.0 ** (7.0 / 12.0)),
    )
    return list(zip(freqs, _PESOS_TRIADE))


def _sintetizar_voz_loop(u: np.ndarray, frequencia: float, timbre: str) -> np.ndarray:
    """Sintetiza voz contínua com número inteiro de ciclos em ``u in [0, 1)``."""
    ciclos = max(1, round(frequencia * DURACAO_BUFFER_LOOP_SEG))
    fase = 2.0 * np.pi * ciclos * u

    if timbre == "piano":
        return (
            0.62 * np.sin(fase)
            + 0.24 * np.sin(2.0 * fase)
            + 0.10 * np.sin(3.0 * fase)
            + 0.04 * np.sin(4.0 * fase)
        )
    if timbre == "violao":
        return 0.68 * np.sin(fase) + 0.22 * np.sin(2.0 * fase) + 0.10 * np.sin(3.0 * fase)
    if timbre == "cordas":
        # Ensemble com micro-desafinação exata (+/- 0.5 Hz em buffer de 2.0s) sem clique
        f_a = 2.0 * np.pi * max(1, ciclos - 1) * u
        f_b = 2.0 * np.pi * (ciclos + 1) * u
        base = 0.44 * np.sin(fase) + 0.28 * np.sin(f_a) + 0.28 * np.sin(f_b)
        return 0.72 * base + 0.20 * np.sin(2.0 * fase) + 0.08 * np.sin(3.0 * fase)
    if timbre == "synth":
        sub = 0.24 * np.sin(2.0 * np.pi * max(1, ciclos // 2) * u)
        serra = (
            0.46 * np.sin(fase)
            + 0.22 * np.sin(2.0 * fase)
            + 0.13 * np.sin(3.0 * fase)
            + 0.08 * np.sin(4.0 * fase)
        )
        return np.tanh(1.25 * (serra + sub)) * 0.85
    if timbre == "8bit":
        bruta = 0.68 * np.sin(fase) + 0.22 * np.sin(3.0 * fase) + 0.10 * np.sin(5.0 * fase)
        return np.round(np.tanh(1.8 * bruta) * 5.0) / 6.0
    return np.sin(fase)


def _sintetizar_plucked(midi: int, timbre: str) -> np.ndarray:
    """Sintetiza nota one-shot expressiva com cauda de reverb estéreo."""
    n = round(TAXA_AMOSTRAGEM * DURACAO_PLUCKED_SEG)
    t = np.arange(n, dtype=np.float64) / TAXA_AMOSTRAGEM
    freq = _midi_para_freq(midi)
    fase = 2.0 * np.pi * freq * t

    if timbre == "violao":
        # Síntese física Karplus-Strong vetorizada por fatias de período
        periodo = max(8, round(TAXA_AMOSTRAGEM / freq))
        rng = np.random.default_rng(midi)
        pulso = rng.uniform(-1.0, 1.0, periodo)
        pulso = 0.5 * (pulso + np.roll(pulso, 1))
        pulso -= pulso.mean()
        onda = np.zeros(n, dtype=np.float64)
        onda[:periodo] = pulso
        for inicio in range(periodo, n, periodo):
            ant = onda[inicio - periodo : inicio]
            filtrado = 0.992 * 0.5 * (ant + np.roll(ant, -1))
            tam = min(periodo, n - inicio)
            onda[inicio : inicio + tam] = filtrado[:tam]
        onda *= 1.15
    elif timbre == "piano":
        env = (0.72 * np.exp(-t * 4.4) + 0.28 * np.exp(-t * 1.6)) * np.minimum(1.0, t / 0.004)
        brilho = np.exp(-t * 16.0)
        onda = (
            0.56 * np.sin(fase)
            + 0.26 * np.sin(2.0 * fase) * (0.35 + 0.65 * brilho)
            + 0.12 * np.sin(3.0 * fase) * brilho
            + 0.06 * np.sin(4.0 * fase) * brilho
        ) * env
    elif timbre == "cordas":
        env = (1.0 - np.exp(-t * 24.0)) * np.exp(-t * 2.2)
        onda = (
            0.52 * np.sin(fase)
            + 0.24 * np.sin(2.0 * np.pi * (freq * 0.997) * t)
            + 0.24 * np.sin(2.0 * np.pi * (freq * 1.003) * t)
            + 0.14 * np.sin(2.0 * fase)
        ) * env
    elif timbre == "synth":
        env = np.minimum(1.0, t / 0.006) * np.exp(-t * 3.2)
        onda = np.tanh(
            1.35 * (
                0.48 * np.sin(fase)
                + 0.24 * np.sin(2.0 * fase)
                + 0.14 * np.sin(3.0 * fase)
                + 0.22 * np.sin(0.5 * fase)
            )
        ) * env
    else:  # 8bit
        env = np.minimum(1.0, t / 0.003) * np.exp(-t * 4.6)
        bruta = 0.68 * np.sin(fase) + 0.22 * np.sin(3.0 * fase) + 0.10 * np.sin(5.0 * fase)
        onda = (np.round(np.tanh(1.8 * bruta) * 5.0) / 6.0) * env

    # Fade-in/out curto antes do reverb para garantir zero cliques
    fade_in = min(n, round(0.003 * TAXA_AMOSTRAGEM))
    fade_out = min(n, round(0.060 * TAXA_AMOSTRAGEM))
    if fade_in > 0:
        onda[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
    if fade_out > 0:
        onda[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)

    estereo = _aplicar_reverb_estereo(onda, circular=False)
    estereo[-fade_out:, :] *= np.linspace(1.0, 0.0, fade_out)[:, None]
    return estereo


def gerar_buffer_onda(
    nota: str,
    escala: TipoEscala,
    timbre: str,
    modo_harmonico: ModoHarmonico,
) -> np.ndarray:
    """Gera matriz estéreo ``(N, 2)`` contínua com reverb circular para testes e mixer."""
    num_amostras = round(TAXA_AMOSTRAGEM * DURACAO_BUFFER_LOOP_SEG)
    u = np.linspace(0.0, 1.0, num_amostras, endpoint=False)
    vozes = _frequencias_para_modo(nota, escala, modo_harmonico)

    soma = np.zeros(num_amostras, dtype=np.float64)
    for freq, peso in vozes:
        soma += peso * _sintetizar_voz_loop(u, freq, timbre)

    return _aplicar_reverb_estereo(np.clip(soma, -1.0, 1.0), circular=True)


def inicializar_audio() -> None:
    """Inicializa o mixer de 16 canais e pré-sintetiza os 5 instrumentos."""
    global _inicializado, _sons_loop, _sons_plucked
    global _canais_loop_vivo, _canais_loop_looper
    global _canais_plucked_vivo, _canais_plucked_looper
    global _idx_loop_vivo, _idx_loop_looper, _idx_plucked_vivo, _idx_plucked_looper

    if _inicializado and pygame.mixer.get_init() is not None:
        return

    if pygame.mixer.get_init() is None:
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

    pygame.mixer.set_num_channels(16)

    # Pré-calcula cada voz MIDI (60..80) com reverb linear circular uma única vez,
    # evitando recalcular senoides repetidas na montagem das 160 combinações.
    num_amostras_loop = round(TAXA_AMOSTRAGEM * DURACAO_BUFFER_LOOP_SEG)
    u_loop = np.linspace(0.0, 1.0, num_amostras_loop, endpoint=False)
    vozes_estereo: dict[tuple[str, int], np.ndarray] = {
        (timbre, midi): _aplicar_reverb_estereo(
            _sintetizar_voz_loop(u_loop, _midi_para_freq(midi), timbre),
            circular=True,
        )
        for timbre in TIMBRES
        for midi in range(60, 81)
    }

    _sons_loop = {}
    for timbre in TIMBRES:
        for modo in MODOS_HARMONICOS:
            for escala in ESCALAS_VALIDAS:
                for nota in NOTAS_FREQUENCIAS:
                    if modo == "nota":
                        midi_n = NOTAS_MIDI[nota] if escala == "maior" else NOTAS_MIDI_MENOR_NATURAL[nota]
                        estereo = vozes_estereo[(timbre, midi_n)]
                    else:
                        midi_base = NOTAS_MIDI[nota]
                        terca = 4 if escala == "maior" else 3
                        estereo = (
                            _PESOS_TRIADE[0] * vozes_estereo[(timbre, midi_base)]
                            + _PESOS_TRIADE[1] * vozes_estereo[(timbre, midi_base + terca)]
                            + _PESOS_TRIADE[2] * vozes_estereo[(timbre, midi_base + 7)]
                        )
                    _sons_loop[(timbre, modo, escala, nota)] = _estereo_para_som(estereo)

    _sons_plucked = {
        (timbre, midi): _estereo_para_som(_sintetizar_plucked(midi, timbre))
        for timbre in TIMBRES
        for midi in range(60, 85)
    }

    _canais_loop_vivo = (pygame.mixer.Channel(0), pygame.mixer.Channel(1))
    _canais_plucked_vivo = [pygame.mixer.Channel(i) for i in range(2, 10)]
    _canais_loop_looper = (pygame.mixer.Channel(10), pygame.mixer.Channel(11))
    _canais_plucked_looper = [pygame.mixer.Channel(i) for i in range(12, 16)]

    _idx_loop_vivo = 0
    _idx_loop_looper = 0
    _idx_plucked_vivo = 0
    _idx_plucked_looper = 0
    _inicializado = True


def _disparar_plucked_no_pool(som: pygame.mixer.Sound, volume: float, canal_looper: bool) -> None:
    """Seleciona canal livre no pool polifônico (ou gira round-robin) e toca."""
    global _idx_plucked_vivo, _idx_plucked_looper

    pool = _canais_plucked_looper if canal_looper else _canais_plucked_vivo
    if not pool:
        return

    escolhido = next((ch for ch in pool if not ch.get_busy()), None)
    if escolhido is None:
        if canal_looper:
            escolhido = pool[_idx_plucked_looper % len(pool)]
            _idx_plucked_looper = (_idx_plucked_looper + 1) % len(pool)
        else:
            escolhido = pool[_idx_plucked_vivo % len(pool)]
            _idx_plucked_vivo = (_idx_plucked_vivo + 1) % len(pool)

    escolhido.set_volume(volume)
    escolhido.play(som)


def definir_timbre(timbre: str | int) -> str:
    """Altera o instrumento ativo (1..5 ou nome) com transição imediata."""
    global _timbre_atual

    if isinstance(timbre, int):
        idx = max(1, min(len(TIMBRES), timbre)) - 1
        novo = TIMBRES[idx]
    else:
        novo = timbre.lower()
        if novo not in TIMBRES:
            raise ValueError(f"Instrumento inválido '{timbre}'. Opções: {TIMBRES}")

    if novo != _timbre_atual:
        _timbre_atual = novo
        if _nota_ativa_vivo is not None and not _arpejo_ativo_vivo:
            tocar_nota(_nota_ativa_vivo, _escala_ativa_vivo, _volume_atual_vivo, arpejo=False)
    return _timbre_atual


def alternar_modo_harmonico() -> ModoHarmonico:
    """Alterna entre ``'acorde'`` (tríade) e ``'nota'`` (nota individual)."""
    global _modo_harmonico_atual

    _modo_harmonico_atual = "nota" if _modo_harmonico_atual == "acorde" else "acorde"
    if _nota_ativa_vivo is not None and not _arpejo_ativo_vivo:
        tocar_nota(_nota_ativa_vivo, _escala_ativa_vivo, _volume_atual_vivo, arpejo=False)
    return _modo_harmonico_atual


def obter_frequencia_referencia(nota: str, escala: TipoEscala = "maior") -> float:
    """Retorna a frequência fundamental em Hz para exibição na HUD."""
    chave = nota.upper()
    if _modo_harmonico_atual == "nota" and escala == "menor":
        return NOTAS_MENOR_NATURAL[chave]
    return NOTAS_FREQUENCIAS[chave]


def obter_estado_audio() -> tuple[str, ModoHarmonico]:
    """Retorna ``(timbre_atual, modo_harmonico_atual)``."""
    return _timbre_atual, _modo_harmonico_atual


def atualizar_volume(volume: float, canal_looper: bool = False) -> None:
    """Atualiza em tempo real o volume dos canais sustentados ativos."""
    global _volume_atual_vivo

    vol = _limitar_volume(volume)
    if not canal_looper:
        _volume_atual_vivo = vol

    if not _inicializado:
        return

    par = _canais_loop_looper if canal_looper else _canais_loop_vivo
    if par is None:
        return
    for ch in par:
        if ch.get_busy():
            ch.set_volume(vol)


def tocar_nota(
    nota: str,
    escala: TipoEscala = "maior",
    volume: float = 0.75,
    arpejo: bool = False,
    timbre_override: str | None = None,
    canal_looper: bool = False,
) -> None:
    """Toca o acorde/nota sustentado (ou prepara o modo arpejo silencioso)."""
    global _idx_loop_vivo, _idx_loop_looper
    global _nota_ativa_vivo, _escala_ativa_vivo, _arpejo_ativo_vivo, _volume_atual_vivo

    inicializar_audio()
    chave = nota.upper()
    if chave not in NOTAS_FREQUENCIAS:
        raise ValueError(f"Nota inválida '{nota}'.")
    if escala not in ESCALAS_VALIDAS:
        raise ValueError(f"Escala inválida '{escala}'.")

    vol = _limitar_volume(volume)
    timbre_efetivo = (timbre_override or _timbre_atual).lower()
    if timbre_efetivo not in TIMBRES:
        timbre_efetivo = _timbre_atual

    if not canal_looper:
        _nota_ativa_vivo = chave
        _escala_ativa_vivo = escala
        _arpejo_ativo_vivo = arpejo
        _volume_atual_vivo = vol

    par = _canais_loop_looper if canal_looper else _canais_loop_vivo
    if par is None:
        return

    if arpejo:
        for ch in par:
            if ch.get_busy():
                ch.fadeout(FADE_OUT_MS)
        return

    idx_atual = _idx_loop_looper if canal_looper else _idx_loop_vivo
    ch_ant = par[idx_atual]
    if ch_ant.get_busy():
        ch_ant.fadeout(FADE_OUT_MS)

    novo_idx = 1 - idx_atual
    if canal_looper:
        _idx_loop_looper = novo_idx
    else:
        _idx_loop_vivo = novo_idx

    ch_novo = par[novo_idx]
    som_loop = _sons_loop[(timbre_efetivo, _modo_harmonico_atual, escala, chave)]
    fator_sust = 0.62 if timbre_efetivo in ("piano", "violao") else 1.0
    ch_novo.set_volume(vol * fator_sust)
    ch_novo.play(som_loop, loops=-1, fade_ms=FADE_IN_MS)

    # Ataque percussivo inicial em Piano e Violão para maior realismo acústico
    if timbre_efetivo in ("piano", "violao"):
        midi_base = NOTAS_MIDI_MENOR_NATURAL[chave] if (_modo_harmonico_atual == "nota" and escala == "menor") else NOTAS_MIDI[chave]
        som_atk = _sons_plucked.get((timbre_efetivo, midi_base))
        if som_atk is not None:
            _disparar_plucked_no_pool(som_atk, vol * 0.65, canal_looper)


def disparar_passo_arpejo(
    nota: str,
    escala: TipoEscala = "maior",
    passo: int = 0,
    volume: float = 0.75,
    timbre_override: str | None = None,
    canal_looper: bool = False,
) -> int:
    """Dispara a próxima nota dedilhada do arpejo e retorna o próximo índice."""
    inicializar_audio()
    chave = nota.upper()
    if chave not in NOTAS_FREQUENCIAS:
        return 0

    vol = _limitar_volume(volume)
    timbre_efetivo = (timbre_override or _timbre_atual).lower()
    if timbre_efetivo not in TIMBRES:
        timbre_efetivo = _timbre_atual

    if _modo_harmonico_atual == "acorde":
        midi_base = NOTAS_MIDI[chave]
        terca = 4 if escala == "maior" else 3
        intervalos = (0, terca, 7, 12)
    else:
        midi_base = NOTAS_MIDI[chave] if escala == "maior" else NOTAS_MIDI_MENOR_NATURAL[chave]
        intervalos = (0, 12)

    idx = passo % len(intervalos)
    midi_alvo = max(60, min(84, midi_base + intervalos[idx]))
    som = _sons_plucked.get((timbre_efetivo, midi_alvo))
    if som is not None:
        _disparar_plucked_no_pool(som, vol, canal_looper)

    return (idx + 1) % len(intervalos)


def parar_nota(canal_looper: bool = False) -> None:
    """Interrompe suavemente o canal sustentado (ao vivo ou do looper)."""
    global _nota_ativa_vivo, _arpejo_ativo_vivo

    if not canal_looper:
        _nota_ativa_vivo = None
        _arpejo_ativo_vivo = False

    if not _inicializado:
        return

    par = _canais_loop_looper if canal_looper else _canais_loop_vivo
    if par is None:
        return
    for ch in par:
        if ch.get_busy():
            ch.fadeout(FADE_OUT_MS)


def encerrar_audio() -> None:
    """Libera os recursos do mixer ao finalizar a aplicação."""
    global _inicializado, _canais_loop_vivo, _canais_loop_looper
    global _nota_ativa_vivo, _arpejo_ativo_vivo

    if _inicializado and pygame.mixer.get_init() is not None:
        parar_nota(canal_looper=False)
        parar_nota(canal_looper=True)
        pygame.mixer.quit()

    _sons_loop.clear()
    _sons_plucked.clear()
    _canais_loop_vivo = None
    _canais_loop_looper = None
    _canais_plucked_vivo.clear()
    _canais_plucked_looper.clear()
    _nota_ativa_vivo = None
    _arpejo_ativo_vivo = False
    _inicializado = False
