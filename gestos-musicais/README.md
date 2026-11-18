# Tocador Musical Gestual (`gestos-musicais`)

Instrumento musical interativo em HD (`1280x720`, janela redimensionável e Tela Cheia) via webcam usando **OpenCV**, **MediaPipe Hands**, **NumPy** e **Pygame Mixer**. Possui **5 instrumentos com reverb estéreo**, **Arpejador espacial com corda luminosa**, **Mini-Looper de base**, **Touch Aéreo** na barra superior e **pop-ups probabilísticos de gatos**.

---

## Pré-requisitos e Execução

```powershell
cd gestos-musicais
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

### Atalhos e Controles na Barra Superior

| Tecla / Botão | Ação |
| :---: | :--- |
| **`1` a `5`** *(ou Touch/Clique no topo)* | Troca instrumento: **1 Piano**, **2 Violão**, **3 Cordas**, **4 Synth**, **5 8-Bit** |
| **`R`** *(ou Touch/Clique em `LOOP`)* | **Mini-Looper**: `Parado → Gravando → Loop Contínuo → Limpar` |
| **`G`** *(ou Touch/Clique em `?`)* | Abre ou fecha o **Guia Visual de Gestos** (oculto por padrão para tela limpa) |
| **`F`** | Alterna entre janela redimensionável e **Tela Cheia** |
| **`H`** | Alterna entre **Acorde (Tríade)** e **Nota Pura** |
| **`ESC` ou `Q`** | Encerra a aplicação |

> **Touch Aéreo:** Aponte a ponta do dedo indicador para qualquer botão da barra superior por `0.40s` para acionar sem usar o teclado!

---

## Controles por Gestos (1 Mão vs. 2 Mãos)

- **1 Mão (Modo Simplificado):** toca apenas as 8 notas (`DO4` a `DO5`) no Modo Maior com volume fixo (`75%`).
- **2 Mãos (Modo Expressivo):**
  - **Mão Esquerda:** escolhe a nota (`DO4` a `DO5`) ou `Silêncio` (punho fechado).
  - **Mão Direita — Altura (`Y`):** controla o **volume** (`10%` a `100%`).
  - **Mão Direita — Postura:** **Palma aberta** (`>= 3` dedos) = **MAIOR** *(verde + gatos felizes)*; **Punho fechado ou Pinça** = **MENOR** *(azul + gatos tristes)* sem cortar a nota.
  - **Aproximar as duas mãos (eixo `X`):** ativa o **Arpejador automático** com uma corda luminosa vibrante entre as palmas (`4.5` a `9.5` notas/s quanto mais próximas).

### Tabela das 8 Notas

| Gesto da Mão | Nota | Gesto da Mão | Nota |
| :--- | :---: | :--- | :---: |
| Apenas **Indicador** | **DO4** | **5 dedos** (Mão aberta) | **SOL4** |
| **Indicador + Médio** ("V") | **RE4** | **Hang Loose** (Polegar + Mindinho) | **LA4** |
| **3 dedos** (Ind + Med + Ane) | **MI4** | **Rock / Chifres** (Ind + Min) | **SI4** |
| **4 dedos** (sem polegar) | **FA4** | **Gesto "L"** (Polegar + Ind) | **DO5** |

---

## Onde Colocar as Imagens dos Gatos (`assets/`)

- `gestos-musicais/assets/happy/` → Gatos felizes (disparados em notas/acordes **Maiores**)
- `gestos-musicais/assets/sad/` → Gatos tristes (disparados em notas/acordes **Menores**)

Os memes disparam de forma probabilística (`40%` por nova nota atacada, cooldown mínimo de `1.5s`).
