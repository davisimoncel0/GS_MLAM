# Global Solution 2026.1 - FIAP
# Modelagem Linear para Aprendizagem de Maquina
# Integrantes:
#   - [NOME COMPLETO] - RM [NUMERO]
#   - [NOME COMPLETO] - RM [NUMERO]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

os.makedirs('graficos', exist_ok=True)

# ===================================
# CARREGAMENTO E PREPARACAO DOS DADOS
# ===================================

df = pd.read_csv('dados/UCS-Satellite-Database.csv', index_col=0)

df.columns = [
    'nome_satelite', 'nome_oficial', 'pais_registro', 'pais_operador',
    'operador', 'usuarios', 'proposito', 'proposito_detalhado',
    'classe_orbita', 'tipo_orbita', 'longitude_geo', 'perigeu_km',
    'apogeu_km', 'excentricidade', 'inclinacao', 'periodo_min',
    'massa_lancamento_kg', 'massa_seca_kg', 'potencia_watts',
    'data_lancamento', 'vida_util_anos', 'fabricante', 'pais_fabricante',
    'local_lancamento', 'veiculo_lancamento', 'cospar', 'norad',
    'comentarios', 'fonte_orbital', 'fonte'
]

for col in ['perigeu_km', 'apogeu_km', 'periodo_min', 'massa_lancamento_kg',
            'potencia_watts', 'vida_util_anos']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['data_lancamento'] = pd.to_datetime(df['data_lancamento'], errors='coerce')
df['ano_lancamento'] = df['data_lancamento'].dt.year.astype('Int64')

print(f'Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas')
print()


# ======================================
# TABELAS DE DISTRIBUICAO DE FREQUENCIAS
# ======================================

# Variável quantitativa discreta: Ano de Lançamento.

print('=' * 55)
print('TABELA DE FREQUÊNCIAS - Ano de Lançamento (discreta)')
print('=' * 55)

anos = df['ano_lancamento'].dropna()
freq_ano = anos.value_counts().sort_index()

tabela_discreta = pd.DataFrame({
    'Ano': freq_ano.index,
    'fi': freq_ano.values
})
tabela_discreta['fri'] = tabela_discreta['fi'] / tabela_discreta['fi'].sum()
tabela_discreta['fri%'] = (tabela_discreta['fri'] * 100).round(2)
tabela_discreta['Fi'] = tabela_discreta['fi'].cumsum()
tabela_discreta['Fri'] = tabela_discreta['fri'].cumsum().round(4)
tabela_discreta.index = range(1, len(tabela_discreta) + 1)

print(f'\nTotal de observações: {len(anos)}')
print(f'Período: {int(anos.min())} a {int(anos.max())}\n')
print(tabela_discreta.to_string())
print()


# Variável quantitativa contínua: Massa de Lançamento (kg).

print('=' * 55)
print('TABELA DE FREQUÊNCIAS - Massa de Lançamento (contínua)')
print('=' * 55)

massa = df['massa_lancamento_kg'].dropna()


n = len(massa)
k = int(np.ceil(1 + 3.322 * np.log10(n)))
h = np.ceil((massa.max() - massa.min()) / k)

print(f'\nTotal de observações: {n}')
print(f'Número de classes (Sturges): {k}')
print(f'Amplitude de classe: {h:.0f} kg')


lim_inf = np.floor(massa.min() / 100) * 100
bins = [lim_inf + i * h for i in range(k + 1)]
if bins[-1] < massa.max():
    bins[-1] = massa.max() + 1

rotulos = [f'[{bins[i]:.0f} - {bins[i+1]:.0f})' for i in range(len(bins) - 1)]
categorias = pd.cut(massa, bins=bins, right=False, labels=rotulos)
freq_massa = categorias.value_counts().sort_index()

tabela_continua = pd.DataFrame({
    'Classe (kg)': freq_massa.index.astype(str),
    'fi': freq_massa.values
})
tabela_continua['fri'] = tabela_continua['fi'] / tabela_continua['fi'].sum()
tabela_continua['fri%'] = (tabela_continua['fri'] * 100).round(2)
tabela_continua['Fi'] = tabela_continua['fi'].cumsum()
tabela_continua['Fri'] = tabela_continua['fri'].cumsum().round(4)
tabela_continua.index = range(1, len(tabela_continua) + 1)

print()
print(tabela_continua.to_string())
print()


# =====================
# GRAFICOS ESTATISTICOS
# =====================

fig, ax = plt.subplots(figsize=(10, 6))

massa_filtrada = massa[massa <= 10000]
ax.hist(massa_filtrada, bins=20, color='steelblue', edgecolor='white')
ax.axvline(massa_filtrada.mean(), color='red', linestyle='--', label=f'Média: {massa_filtrada.mean():.0f} kg')
ax.axvline(massa_filtrada.median(), color='green', linestyle='--', label=f'Mediana: {massa_filtrada.median():.0f} kg')

ax.set_title('Distribuição da Massa de Lançamento dos Satélites')
ax.set_xlabel('Massa de Lançamento (kg)')
ax.set_ylabel('Frequência')
ax.legend()

plt.tight_layout()
plt.savefig('graficos/grafico_01_histograma_massa.png', dpi=150)
plt.close()
print('Gráfico 1 salvo: graficos/grafico_01_histograma_massa.png')


fig, ax = plt.subplots(figsize=(10, 6))

top_paises = df['pais_operador'].value_counts().head(10)
cores = plt.cm.tab10(range(len(top_paises)))

ax.barh(top_paises.index[::-1], top_paises.values[::-1], color=cores)
for i, v in enumerate(top_paises.values[::-1]):
    ax.text(v + 2, i, str(v), va='center')

ax.set_title('Top 10 Paises por Número de Satélites em Operação')
ax.set_xlabel('Quantidade de Satélites')
ax.set_ylabel('País')

plt.tight_layout()
plt.savefig('graficos/grafico_02_satelites_por_pais.png', dpi=150)
plt.close()
print('Gráfico 2 salvo: graficos/grafico_02_satelites_por_pais.png')
print()


# ====================
# ANALISES UNIVARIADAS
# ====================

def analise_descritiva(dados, nome, unidade):
    dados = dados.dropna()

    media = dados.mean()
    mediana = dados.median()
    moda_res = stats.mode(dados, keepdims=True)
    moda = moda_res.mode[0]

    maximo = dados.max()
    minimo = dados.min()
    amplitude = maximo - minimo
    variancia = dados.var(ddof=1)
    desvio = dados.std(ddof=1)

    q1 = dados.quantile(0.25)
    q2 = dados.quantile(0.50)
    q3 = dados.quantile(0.75)

    print(f'Variável: {nome}')
    print(f'Observações: {len(dados)}')
    print()
    print('Medidas de Tendência Central:')
    print(f'  Média:   {media:,.2f} {unidade}')
    print(f'  Mediana: {mediana:,.2f} {unidade}')
    print(f'  Moda:    {moda:,.2f} {unidade}')
    print()
    print('Medidas de Dispersão:')
    print(f'  Maximo:        {maximo:,.2f} {unidade}')
    print(f'  Minimo:        {minimo:,.2f} {unidade}')
    print(f'  Amplitude:     {amplitude:,.2f} {unidade}')
    print(f'  Variancia:     {variancia:,.2f}')
    print(f'  Desvio Padrao: {desvio:,.2f} {unidade}')
    print()
    print('Quartis:')
    print(f'  Q1 (25%): {q1:,.2f} {unidade}')
    print(f'  Q2 (50%): {q2:,.2f} {unidade}')
    print(f'  Q3 (75%): {q3:,.2f} {unidade}')

    return {
        'media': media, 'mediana': mediana, 'moda': moda,
        'maximo': maximo, 'minimo': minimo, 'amplitude': amplitude,
        'variancia': variancia, 'desvio': desvio,
        'q1': q1, 'q2': q2, 'q3': q3
    }



print('=' * 55)
print('ANALISE UNIVARIADA 1 - Massa de Lancamento (kg)')
print('=' * 55)
print()
r1 = analise_descritiva(df['massa_lancamento_kg'], 'Massa de Lancamento', 'kg')
print()
print('Interpretacao:')
print(f'  A massa de lancamento tem alta variabilidade, com media de {r1["media"]:,.0f} kg')
print(f'  e mediana de {r1["mediana"]:,.0f} kg. A diferenca entre media e mediana mostra')
print(f'  que a distribuicao e assimetrica positiva, ou seja, existem satelites')
print(f'  muito pesados que puxam a media pra cima. A maioria dos satelites tem')
print(f'  massa abaixo de {r1["q3"]:,.0f} kg (Q3).')
print()
print()


print('=' * 55)
print('ANALISE UNIVARIADA 2 - Periodo Orbital (min)')
print('=' * 55)
print()
r2 = analise_descritiva(df['periodo_min'], 'Periodo Orbital', 'min')
print()
print('Interpretacao:')
print(f'  O periodo orbital varia muito ({r2["minimo"]:,.1f} a {r2["maximo"]:,.1f} min).')
print(f'  Periodos perto de 95 min sao orbitas baixas (LEO) e perto de 1436 min')
print(f'  (~24h) sao orbitas geostacionarias (GEO). A mediana de {r2["mediana"]:,.1f} min')
print(f'  indica que a maioria dos satelites esta em orbitas mais baixas.')
print()