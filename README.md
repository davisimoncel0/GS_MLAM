# Global Solution — MLAM | FIAP 2026
## Análise Estatística: All Space Missions from 1957

### Disciplina
Modelagem Linear para Aprendizagem de Máquina — Ciência da Computação | Turma 1CC

---

## Descrição

Análise estatística exploratória completa sobre **4.324 missões espaciais reais** realizadas entre 1957 e 2020, abrangendo 56 organizações de todo o mundo.

**Fonte dos dados:** [All Space Missions from 1957 — Kaggle](https://www.kaggle.com/datasets/agirlcoding/all-space-missions-from-1957)

Utilizamos: Python 3, pandas, numpy, matplotlib e reportlab.

---

## Estrutura do Repositório

```
GS_MLAM/
├── analise.py         
├── README.md                
├── relatorio_estatistico.pdf 
└── Space_Corrected.csv
```

---

## O que foi feito

| Requisito | Descrição |
|-----------|-----------|
| 01 — Base de dados | Dataset real do Kaggle — 4.324 missões (1957–2020) |
| 02a — Tabela discreta | Distribuição de frequências de `ano` (por década) |
| 02b — Tabela contínua | Distribuição de frequências de `custo_milhoes` (Sturges) |
| 03a — Gráfico 1 | Barras — Missões por década |
| 03b — Gráfico 2 | Barras horizontais — Taxa de sucesso por empresa (Top 8) |
| 04a — Univariada 1 | Estatística descritiva completa de `custo_milhoes` (n=949) |
| 04b — Univariada 2 | Estatística descritiva completa de `ano` (n=4.324) |
| 05 — Relatório PDF | Documento técnico consolidado com interpretações e insights |

---

## Principais Resultados

- **4.324 missões** analisadas de **56 organizações** entre **1957 e 2020**
- **89,7%** de taxa geral de sucesso nas missões
- **Pico histórico** na década de 1970 (1.012 missões — Guerra Fria)
- **60,8%** das missões custam até US$ 86 milhões
- Custo médio: US$ 129,8 mi | Mediana: US$ 62 mi (forte assimetria positiva)

---

## Como Executar

```bash
pip install pandas numpy matplotlib reportlab
python3 analise.py
```

---
