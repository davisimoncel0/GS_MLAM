#  GLOBAL SOLUTION — MLAM | FIAP — Ciência da Computação
#  Dataset: All Space Missions from 1957 (Kaggle)
#  Link: kaggle.com/datasets/agirlcoding/all-space-missions-from-1957

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 130
plt.rcParams['font.family'] = 'DejaVu Sans'

# LEITURA E LIMPEZA
df_raw = pd.read_csv('Space_Corrected.csv')

# Renomear colunas
df_raw.columns = ['idx1','idx2','empresa','localizacao','data','missao','status_foguete','custo_milhoes','resultado']

# Extrair ano
df_raw['ano'] = df_raw['data'].str.extract(r'(\d{4})').astype(float)

# Limpar custo (é string com possível espaço)
df_raw['custo_milhoes'] = pd.to_numeric(df_raw['custo_milhoes'], errors='coerce')

# Dataset final — apenas colunas relevantes, sem nulos nas variáveis principais
df = df_raw[['empresa','ano','missao','status_foguete','custo_milhoes','resultado']].copy()
df = df.dropna(subset=['ano'])
df['ano'] = df['ano'].astype(int)

print("=" * 60)
print("DATASET: All Space Missions from 1957 (Kaggle)")
print(f"Total de registros: {len(df)}")
print("=" * 60)
print(df[['empresa','ano','custo_milhoes','resultado']].describe(include='all').to_string())


# QUESTÃO 02 — Tabelas de Distribuição de Frequências

# 2a. DISCRETA: Ano de lançamento (agrupado por década)
# Usamos missões por ano como variável discreta
print("\n" + "=" * 60)
print("TABELA 1 — Distribuição de Frequências: Ano de Lançamento")
print("(Variável Quantitativa Discreta — agrupado por década)")
print("=" * 60)

df['decada'] = (df['ano'] // 10) * 10
freq_dec = df['decada'].value_counts().sort_index()
fr_dec   = (freq_dec / len(df) * 100).round(2)
fi_dec   = freq_dec.cumsum()
fra_dec  = fr_dec.cumsum().round(2)

tabela_discreta = pd.DataFrame({
    'Década': [f"{d}s" for d in freq_dec.index],
    'fi': freq_dec.values,
    'fr (%)': fr_dec.values,
    'Fi': fi_dec.values,
    'Fra (%)': fra_dec.values
})
print(tabela_discreta.to_string(index=False))

# 2b. CONTÍNUA: Custo da missão (milhões USD) 
print("\n" + "=" * 60)
print("TABELA 2 — Distribuição de Frequências: Custo (milhões USD)")
print("(Variável Quantitativa Contínua — Método de Sturges)")
print("=" * 60)

df_custo = df.dropna(subset=['custo_milhoes']).copy()
n_c = len(df_custo)
k   = int(np.ceil(1 + 3.322 * np.log10(n_c)))
mn, mx = df_custo['custo_milhoes'].min(), df_custo['custo_milhoes'].max()
h  = (mx - mn) / k
bins = [round(mn + i * h, 1) for i in range(k + 1)]
labels = [f"[{bins[i]:.1f} ├ {bins[i+1]:.1f})" for i in range(k)]

df_custo['classe'] = pd.cut(df_custo['custo_milhoes'], bins=bins,
                             labels=labels, right=False, include_lowest=True)
fc   = df_custo['classe'].value_counts().sort_index()
frc  = (fc / n_c * 100).round(2)
fic  = fc.cumsum()
frac = frc.cumsum().round(2)
mp   = [round((bins[i] + bins[i+1]) / 2, 1) for i in range(k)]

tabela_continua = pd.DataFrame({
    'Classe (mi USD)': labels,
    'Ponto Médio': mp,
    'fi': fc.values,
    'fr (%)': frc.values,
    'Fi': fic.values,
    'Fra (%)': frac.values
})
print(tabela_continua.to_string(index=False))


# QUESTÃO 03 — Gráficos

# Gráfico 1: Missões por década (barras)
fig1, ax1 = plt.subplots(figsize=(11, 6))
decadas  = [f"{d}s" for d in freq_dec.index]
valores  = freq_dec.values
palette  = ['#1f4e79','#2e75b6','#2e75b6','#4472c4','#5b9bd5',
            '#9dc3e6','#4472c4','#2e75b6','#c55a11']

bars = ax1.bar(decadas, valores, color=palette[:len(decadas)],
               edgecolor='white', linewidth=0.8, width=0.65)

for bar, v in zip(bars, valores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(v), ha='center', va='bottom', fontsize=10, fontweight='bold',
             color='#1f4e79')

ax1.set_title('Número de Missões Espaciais por Década (1957–2020)',
              fontsize=14, fontweight='bold', pad=14)
ax1.set_xlabel('Década', fontsize=12)
ax1.set_ylabel('Número de Missões', fontsize=12)
ax1.set_ylim(0, max(valores) * 1.15)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('grafico1_missoes_por_decada.png', bbox_inches='tight')
plt.close()
print("\n[OK] Gráfico 1 salvo.")

# Gráfico 2: Taxa de sucesso por empresa (top 8)
fig2, ax2 = plt.subplots(figsize=(11, 6))
top8 = df['empresa'].value_counts().head(8).index.tolist()
df_top = df[df['empresa'].isin(top8)].copy()
df_top['sucesso'] = df_top['resultado'].apply(
    lambda x: 1 if x == 'Success' else 0)

taxa = df_top.groupby('empresa')['sucesso'].mean() * 100
taxa_sorted = taxa.sort_values(ascending=True)

cores_barra = ['#c55a11' if v < 80 else '#2e75b6' for v in taxa_sorted.values]
barras = ax2.barh(taxa_sorted.index, taxa_sorted.values,
                  color=cores_barra, edgecolor='white', height=0.65)

for bar, v in zip(barras, taxa_sorted.values):
    ax2.text(v + 0.5, bar.get_y() + bar.get_height()/2,
             f'{v:.1f}%', va='center', fontsize=10, fontweight='bold')

ax2.set_title('Taxa de Sucesso das Missões por Empresa (Top 8)',
              fontsize=14, fontweight='bold', pad=14)
ax2.set_xlabel('Taxa de Sucesso (%)', fontsize=12)
ax2.set_ylabel('Empresa', fontsize=12)
ax2.set_xlim(0, 115)
ax2.axvline(100, color='#aaaaaa', linestyle='--', linewidth=0.8)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2e75b6', label='≥ 80% de sucesso'),
                   Patch(facecolor='#c55a11', label='< 80% de sucesso')]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('grafico2_taxa_sucesso.png', bbox_inches='tight')
plt.close()
print("[OK] Gráfico 2 salvo.")


# QUESTÃO 04 — Análises Univariadas

def analise_univariada(serie, nome_var, unidade=''):
    s = serie.dropna()
    media     = s.mean()
    mediana   = s.median()
    moda_v    = s.mode()[0]
    maximo    = s.max()
    minimo    = s.min()
    amplitude = maximo - minimo
    variancia = s.var(ddof=1)
    desvio    = s.std(ddof=1)
    q1 = s.quantile(0.25)
    q2 = s.quantile(0.50)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    print(f"\n{'='*60}")
    print(f"ANÁLISE UNIVARIADA — {nome_var}  (n={len(s)})")
    print(f"{'='*60}")
    print(f"  Medidas de Tendência Central")
    print(f"    Média      : {media:.2f} {unidade}")
    print(f"    Mediana    : {mediana:.2f} {unidade}")
    print(f"    Moda       : {moda_v:.2f} {unidade}")
    print(f"  Medidas de Dispersão")
    print(f"    Mínimo     : {minimo:.2f} {unidade}")
    print(f"    Máximo     : {maximo:.2f} {unidade}")
    print(f"    Amplitude  : {amplitude:.2f} {unidade}")
    print(f"    Variância  : {variancia:.2f}")
    print(f"    Desvio Pad.: {desvio:.2f} {unidade}")
    print(f"  Medidas Separatrizes")
    print(f"    Q1 (25%)   : {q1:.2f} {unidade}")
    print(f"    Q2 (50%)   : {q2:.2f} {unidade}")
    print(f"    Q3 (75%)   : {q3:.2f} {unidade}")
    print(f"    IIQ        : {iqr:.2f} {unidade}")
    return dict(media=media, mediana=mediana, moda=moda_v,
                maximo=maximo, minimo=minimo, amplitude=amplitude,
                variancia=variancia, desvio=desvio, q1=q1, q2=q2, q3=q3, iqr=iqr)

stats_custo = analise_univariada(df['custo_milhoes'], 'Custo da Missão', 'mi USD')
stats_ano   = analise_univariada(df['ano'].astype(float), 'Ano de Lançamento', '')

print("\n[OK] Análises univariadas concluídas.")

# Salvar stats para o relatório
import json
resumo = {
    'total_missoes': len(df),
    'empresas_unicas': df['empresa'].nunique(),
    'anos': [int(df['ano'].min()), int(df['ano'].max())],
    'taxa_sucesso_geral': round(df[df['resultado']=='Success'].shape[0] / len(df) * 100, 1),
    'custo': stats_custo,
    'ano_stats': stats_ano,
    'decadas': tabela_discreta.to_dict('records'),
}
with open('stats.json', 'w') as f:
    json.dump(resumo, f, default=str)
print("[OK] Stats exportados para stats.json")
