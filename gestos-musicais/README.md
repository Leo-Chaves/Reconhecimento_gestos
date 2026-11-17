# Theremin Gestual (`gestos-musicais`)

Instrumento musical em tempo real via webcam usando **OpenCV**, **MediaPipe Hands**, **NumPy** e **Pygame Mixer**. Cada gesto da mão toca notas da escala de **Dó Maior** sintetizadas em memória (sem arquivos `.wav` externos), enquanto a altura do punho controla o volume e um avaliador musical reage às suas sequências exibindo **memes animados** na tela!

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

```powershell
cd gestos-musicais
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(Caso o PowerShell bloqueie o script de ativação na primeira vez, libere para o usuário atual com `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.)*

---

## Execução e Atalhos ao Vivo

```powershell
python main.py
```

| Tecla | Ação |
| :---: | :--- |
| **`1` / `2` / `3`** | Troca de timbre: **1 = Theremin Clássico (vibrato)**, **2 = Flauta Synth**, **3 = Retrô 8-bit** |
| **`M`** | Alterna entre **Modo 1 Mão** (padrão) e **Modo 2 Mãos** |
| **`C`** | Limpa a trilha de notas recentes |
| **`ESC` ou `Q`** | Encerra a aplicação |

---

## Tabela de Gestos (Escala de Dó Maior)

| Gesto da Mão | Nota | Frequência Base (Hz) | Observação |
| :--- | :---: | :---: | :--- |
| Apenas **Indicador** levantado | **DO** | 261.63 Hz | 1 dedo |
| **Indicador + Médio** levantados | **RE** | 293.66 Hz | 2 dedos |
| **Indicador + Médio + Anelar** levantados | **MI** | 329.63 Hz | 3 dedos |
| **4 dedos** levantados (sem polegar) | **FA** | 349.23 Hz | Polegar recolhido na palma |
| **Mão totalmente aberta** (5 dedos) | **SOL** | 392.00 Hz | Todos os dedos estendidos |
| **Pinça** (polegar `[4]` + indicador `[8]`) | **LA** | 440.00 Hz | Distância normalizada `< 0.05` |
| **Indicador + Mindinho** (gesto Rock) | **SI** | 493.88 Hz | 🤘 |
| **Punho fechado** (nenhum dedo aberto) | **Silêncio** | — | Interrompe a nota suavemente |

> **Modo 2 Mãos (tecla `M`):** A **mão direita** continua escolhendo a nota, enquanto a **mão esquerda** controla o **volume pela altura do punho** e a **oitava** pela quantidade de dedos abertos (`0–1` dedo = Grave `-1`, `2–3` dedos = Normal `0`, `4–5` dedos = Aguda `+1`).

---

## Controle Dinâmico de Volume pela Altura do Braço

1. **Coordenada do Punho:** Lê `landmarks[0].y` (`0.0` topo → `1.0` base).
2. **Clamping e Mapeamento Linear:** Faixa útil de `y = 0.85` (**10%**) até `y = 0.15` (**100%**).
3. **Suavização Exponencial (EMA):** `vol_suave = 0.7 * vol_suave + 0.3 * vol_bruto` evita oscilações bruscas por tremores da mão.
4. **VU Meter:** A barra lateral acompanha em tempo real com gradiente **Verde → Amarelo → Vermelho**.

---

## Avaliador Musical e Memes (`assets/memes/`)

Em **Modo Livre**, o módulo `music_evaluator.py` analisa as últimas 4–5 notas tocadas:
- **Sequência Boa** (escalas como `DO → RE → MI → FA`, arpejos como `DO → MI → SOL`, melodias variadas) → Pop-up de **animal fofo** + fanfarra alegre!
- **Sequência Ruim** (repetições robóticas como `DO → RE → DO → RE`, trítono `FA ↔ SI` ou saltos caóticos) → Pop-up de **gatinho triste** + efeito *"wah-wah"*!

Você pode adicionar ou trocar as imagens livremente colocando arquivos `.png`, `.jpg` ou `.webp` nas pastas:
- `assets/memes/good/`
- `assets/memes/bad/`
