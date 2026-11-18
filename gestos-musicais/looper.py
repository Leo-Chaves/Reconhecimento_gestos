"""Mini-Looper musical para gravar e repetir bases em segundo plano.

Permite gravar uma sequência de notas, acordes ou passos de arpejo com seu
respectivo instrumento original e reproduzi-la em loop pelos canais dedicados
do ``audio_engine`` enquanto você troca de timbre para solar por cima.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import audio_engine
from audio_engine import TipoEscala

EstadoLooper = Literal["parado", "gravando", "tocando"]
DURACAO_MINIMA_LOOP_SEG: float = 0.35
FATOR_VOLUME_BASE_LOOP: float = 0.68


@dataclass(frozen=True)
class EventoLoop:
    """Evento temporal gravado dentro de um ciclo do Mini-Looper."""

    tempo_rel: float
    tipo: Literal["nota", "arpejo", "parar"]
    nota: str | None = None
    escala: TipoEscala = "maior"
    timbre: str = "piano"
    volume: float = 0.75
    arpejo: bool = False
    passo: int = 0
    padrao_arpejo: str = "sobe"


class MiniLooper:
    """Controla gravação e reprodução cíclica não-bloqueante de uma base."""

    def __init__(self) -> None:
        self.estado: EstadoLooper = "parado"
        self.duracao: float = 0.0
        self._eventos: list[EventoLoop] = []
        self._inicio_gravacao: float = 0.0
        self._inicio_ciclo: float = 0.0
        self._idx_playback: int = 0

    def contar_eventos(self) -> int:
        """Retorna a quantidade de eventos gravados no loop atual."""
        return len(self._eventos)

    def parar(self) -> None:
        """Interrompe reprodução/gravação e limpa os eventos do looper."""
        self.estado = "parado"
        self.duracao = 0.0
        self._eventos.clear()
        self._idx_playback = 0
        audio_engine.parar_nota(canal_looper=True)

    def alternar(
        self,
        agora: float,
        nota_atual: str | None = None,
        escala_atual: TipoEscala = "maior",
        timbre_atual: str = "piano",
        volume_atual: float = 0.75,
        arpejo_atual: bool = False,
    ) -> EstadoLooper:
        """Cicla entre ``parado -> gravando -> tocando -> parado``."""
        if self.estado == "parado":
            self.estado = "gravando"
            self.duracao = 0.0
            self._eventos.clear()
            self._inicio_gravacao = agora
            self._idx_playback = 0
            if nota_atual is not None and not arpejo_atual:
                self._eventos.append(
                    EventoLoop(
                        tempo_rel=0.0,
                        tipo="nota",
                        nota=nota_atual,
                        escala=escala_atual,
                        timbre=timbre_atual,
                        volume=volume_atual,
                        arpejo=False,
                    )
                )
            return self.estado

        if self.estado == "gravando":
            dur = agora - self._inicio_gravacao
            if dur >= DURACAO_MINIMA_LOOP_SEG and self._eventos:
                self.duracao = dur
                self.estado = "tocando"
                self._inicio_ciclo = agora
                self._idx_playback = 0
                self.atualizar(agora)
                return self.estado
            self.parar()
            return self.estado

        self.parar()
        return self.estado

    def registrar_nota(
        self,
        nota: str,
        escala: TipoEscala,
        timbre: str,
        volume: float,
        agora: float,
    ) -> None:
        """Registra ataque de acorde/nota sustentada durante a gravação."""
        if self.estado != "gravando":
            return
        self._eventos.append(
            EventoLoop(
                tempo_rel=max(0.0, agora - self._inicio_gravacao),
                tipo="nota",
                nota=nota,
                escala=escala,
                timbre=timbre,
                volume=volume,
                arpejo=False,
            )
        )

    def registrar_passo_arpejo(
        self,
        nota: str,
        escala: TipoEscala,
        timbre: str,
        passo: int,
        volume: float,
        agora: float,
        padrao_arpejo: str = "sobe",
    ) -> None:
        """Registra um passo dedilhado do arpejador durante a gravação."""
        if self.estado != "gravando":
            return
        self._eventos.append(
            EventoLoop(
                tempo_rel=max(0.0, agora - self._inicio_gravacao),
                tipo="arpejo",
                nota=nota,
                escala=escala,
                timbre=timbre,
                volume=volume,
                arpejo=True,
                passo=passo,
                padrao_arpejo=padrao_arpejo,
            )
        )

    def registrar_parada(self, agora: float) -> None:
        """Registra entrada em silêncio durante a gravação."""
        if self.estado != "gravando":
            return
        self._eventos.append(
            EventoLoop(
                tempo_rel=max(0.0, agora - self._inicio_gravacao),
                tipo="parar",
            )
        )

    def _executar_evento(self, ev: EventoLoop) -> None:
        vol_base = max(0.08, ev.volume * FATOR_VOLUME_BASE_LOOP)
        if ev.tipo == "parar" or ev.nota is None:
            audio_engine.parar_nota(canal_looper=True)
            return
        if ev.tipo == "arpejo":
            audio_engine.parar_nota(canal_looper=True)
            audio_engine.disparar_passo_arpejo(
                ev.nota,
                escala=ev.escala,
                passo=ev.passo,
                volume=vol_base,
                timbre_override=ev.timbre,
                canal_looper=True,
                padrao_override=ev.padrao_arpejo,
            )
            return
        audio_engine.tocar_nota(
            ev.nota,
            escala=ev.escala,
            volume=vol_base,
            arpejo=False,
            timbre_override=ev.timbre,
            canal_looper=True,
        )

    def atualizar(self, agora: float) -> None:
        """Avança o cursor do loop e dispara eventos pendentes em tempo real."""
        if self.estado != "tocando" or self.duracao <= 0.0 or not self._eventos:
            return

        decorrido = agora - self._inicio_ciclo
        if decorrido >= self.duracao:
            ciclos_passados = int(decorrido // self.duracao)
            self._inicio_ciclo += ciclos_passados * self.duracao
            self._idx_playback = 0
            audio_engine.parar_nota(canal_looper=True)
            decorrido = max(0.0, agora - self._inicio_ciclo)

        while self._idx_playback < len(self._eventos):
            ev = self._eventos[self._idx_playback]
            if ev.tempo_rel > decorrido:
                break
            self._executar_evento(ev)
            self._idx_playback += 1

    def obter_progresso(self, agora: float) -> float:
        """Retorna segundos gravados (em ``gravando``) ou fração ``0..1`` (em ``tocando``)."""
        if self.estado == "gravando":
            return max(0.0, agora - self._inicio_gravacao)
        if self.estado == "tocando" and self.duracao > 0.0:
            return max(0.0, min(1.0, (agora - self._inicio_ciclo) / self.duracao))
        return 0.0
