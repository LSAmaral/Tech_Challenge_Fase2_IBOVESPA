import pandas as pd

# Carrega o arquivo limpo que criamos
print("Carregando arquivo...")
df = pd.read_csv("ibovespa_preparado.csv")

# Garante que a coluna de data é entendida como tempo, não texto
df["data"] = pd.to_datetime(df["data"])

# Ordena do mais antigo para o mais novo (vital para médias móveis!)
df = df.sort_values(by="data", ascending=True)

print("\n--- 5 Primeiras Linhas (Head) ---")
print(df.head())

print("\n--- Informações dos Tipos (Info) ---")
print(df.info())
