import pandas as pd

# 1. Carregar os dados
print("Carregando dados limpos...")
df = pd.read_csv("ibovespa_preparado.csv")
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values(by="data", ascending=True).reset_index(drop=True)

# 2. Criar Features (Variáveis que ajudam a prever)
print("Criando indicadores técnicos...")

# Média Móvel Simples (SMA) de 7 e 21 dias
# .rolling(7).mean() pega a janela de 7 linhas e tira a média
df["media_movel_7"] = df["fechamento"].rolling(window=7).mean()
df["media_movel_21"] = df["fechamento"].rolling(window=21).mean()

# 3. Criar o Target (O que queremos prever)
# .shift(-1) cria uma coluna "espiando" o preço de amanhã trazido para a linha de hoje
df["fechamento_amanha"] = df["fechamento"].shift(-1)

# Lógica do Alvo:
# Se amanhã for maior que hoje -> 1 (Alta)
# Senão -> 0 (Baixa)
# astype(int) transforma True/False em 1/0
df["target"] = (df["fechamento_amanha"] > df["fechamento"]).astype(int)

# 4. Limpeza Final (Remover linhas vazias geradas pelas médias e shift)
# As primeiras 20 linhas não terão média de 21 dias (NaN)
# A última linha não terá "amanhã" (NaN)
df = df.dropna()

print("\n--- Dados Prontos para Machine Learning ---")
print(df[["data", "fechamento", "media_movel_7", "target"]].tail(10))

print("\n--- Estatísticas do Target ---")
print(df["target"].value_counts(normalize=True))
