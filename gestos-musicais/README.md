# Theremin Gestual (`gestos-musicais`)

Instrumento musical em tempo real via webcam usando **OpenCV**, **MediaPipe Hands**, **NumPy** e **Pygame Mixer**. Cada gesto da mão dispara uma nota da escala de **Dó Maior** sintetizada em memória (sem arquivos `.wav` externos), enquanto a altura do braço/punho controla continuamente o volume sonoro.

---

## Pré-requisitos

- **Windows** (PowerShell)
- **Python 3.12** (necessário para compatibilidade oficial dos *wheels* do `mediapipe==0.10.21` com `mediapipe.solutions.hands` e `pygame`)
- Webcam conectada e saída de áudio ativa

> **Nota sobre o Python 3.12:** Caso tenha apenas Python 3.14 instalado pelo *Python Install Manager*, instale o 3.12 lado a lado sem remover o 3.14:
> ```powershell
> py install 3.12
> ```

---

## Configuração e Ativação da `.venv` (PowerShell)

Entre na pasta `gestos-musicais`, crie o ambiente virtual isolado com Python 3.12, ative-o e instale as dependências:

```powershell
cd gestos-musicais
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(Caso o PowerShell bloqueie o script de ativação na primeira vez, libere para o usuário atual com `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.)*

---

## Execução

Com a `.venv` ativada:

```powershell
python main.py
```

Para encerrar o programa a qualquer momento, pressione **`ESC`** ou **`q`**.

---

## Tabela de Gestos (Escala de Dó Maior)

| Gesto da Mão | Nota | Frequência (Hz) | Observação |
| :--- | :---: | :---: | :--- |
| Apenas **Indicador** levantado | **DO** | 261.63 Hz | 1 dedo |
| **Indicador + Médio** levantados | **RE** | 293.66 Hz | 2 dedos |
| **Indicador + Médio + Anelar** levantados | **MI** | 329.63 Hz | 3 dedos |
| **4 dedos** levantados (sem polegar) | **FA** | 349.23 Hz | Polegar recolhido na palma |
| **Mão totalmente aberta** (5 dedos) | **SOL** | 392.00 Hz | Todos os dedos estendidos |
| **Pinça** (polegar `[4]` + indicador `[8]`) | **LA** | 440.00 Hz | Distância normalizada `< 0.05` |
| **Indicador + Mindinho** (gesto Rock) | **SI** | 493.88 Hz | 🤘 |
| **Punho fechado** (nenhum dedo aberto) | **Silêncio** | — | Interrompe a nota suavemente |

> **Estabilidade (Debounce):** Toda troca de gesto só é confirmada após **3 frames consecutivos** idênticos, evitando disparos acidentais durante transições rápidas entre dedos.

---

## Controle Dinâmico de Volume pela Altura do Braço

O volume é modulado continuamente enquanto a nota está soando:

1. **Captura da Altura:** O sistema lê a coordenada vertical normalizada do punho (`landmarks[0].y`, onde `0.0` é o topo da câmera e `1.0` é a base).
2. **Faixa Útil e Clamping:** A região útil vai de `y = 0.85` (braço baixo → **10% / 0.10**) até `y = 0.15` (braço alto → **100% / 1.00**).
3. **Suavização Exponencial (EMA):** Aplica-se `vol_suave = 0.7 * vol_suave + 0.3 * vol_bruto` a cada frame, eliminando variações bruscas causadas por tremores naturais da mão.
4. **Modulação e VU Meter:** O volume calculado é aplicado em tempo real via `audio_engine.atualizar_volume(vol_suave)` e refletido na barra vertical **VU Meter** na lateral direita da tela (transição de **Verde → Amarelo → Vermelho**).
