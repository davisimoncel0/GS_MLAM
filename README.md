# Global Solution 2026.1 - FIAP

## Modelagem Linear para Aprendizagem de Maquina

Análise estatística descritiva do UCS Satellite Database.

## Integrantes

- [Davi Simoncelo] - RM [571738]
- [João Pedro Sousa] - RM [573962]
- [Matheus Evangelista Silva] - RM [568593]

Turma: 1CCPK
Curso: Ciência da Computação

## Descrição

O projeto aplica conceitos de estatística descritiva para analisar dados reais de satélites em órbita, usando a base UCS Satellite Database. A base contém dados de 576 satélites operacionais com informações como massa, período orbital, país de operação, entre outros.

O projeto inclui:
- Tabelas de distribuição de frequências (variável discreta e contínua)
- Gráficos estatísticos (histograma e gráfico de barras)
- Análises univariadas com medidas de tendência central, dispersão e quartis
- Relatório estatístico em PDF

# Fonte dos Dados

UCS Satellite Database
https://www.kaggle.com/datasets/mexwell/ucs-satellite-database

Disponível da mesma forma em: https://www.ucs.org/resources/satellite-database

# Como Executar

Instalar dependências:
```
pip install pandas numpy matplotlib seaborn scipy fpdf2
```

Rodar o script:
```
python3 analise.py
```