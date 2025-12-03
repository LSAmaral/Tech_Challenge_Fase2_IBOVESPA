import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --- 1. FUNÇÃO AUXILIAR: RSI ---
def calcular_rsi(series, janela=14):
    delta = series.diff()
    ganho = (delta.where(delta > 0, 0)).rolling(window=janela).mean()
    perda = (-delta.where(delta < 0, 0)).rolling(window=janela).mean()

    rs = ganho / perda
    return 100 - (100 / (1 + rs))


# --- 2. CARGA E PREPARAÇÃO ---
print("⚙️ Carregando dados...")
df = pd.read_csv("ibovespa_preparado.csv")
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values(by="data", ascending=True).reset_index(drop=True)

# --- 3. ENGENHARIA DE ATRIBUTOS (TURBINADA) ---
# Retorno diário (%)
df["retorno"] = df["fechamento"].pct_change()

# Médias Móveis (Tendência)
df["media_7"] = df["fechamento"].rolling(7).mean()
df["media_21"] = df["fechamento"].rolling(21).mean()

# RSI (O Salvador da Pátria) - Detecta se caiu demais (sobrevendido)
df["rsi_14"] = calcular_rsi(df["fechamento"], janela=14)

# Momentum (Preço hoje vs Preço de 5 dias atrás)
df["momentum"] = df["fechamento"] / df["fechamento"].shift(5)

# Target
df["fechamento_amanha"] = df["fechamento"].shift(-1)
df["target"] = (df["fechamento_amanha"] > df["fechamento"]).astype(int)

df = df.dropna()

# --- 4. DIVISÃO TREINO vs TESTE ---
dias_teste = 30
indice_corte = len(df) - dias_teste

# Adicionamos 'rsi_14' e 'momentum' nas features
features = ["fechamento", "media_7", "media_21", "rsi_14", "momentum"]
X = df[features]
y = df["target"]

X_treino = X.iloc[:indice_corte]
y_treino = y.iloc[:indice_corte]
X_teste = X.iloc[indice_corte:]
y_teste = y.iloc[indice_corte:]

# --- 5. TREINAMENTO ---
print(f"🤖 Treinando com {len(X_treino)} dias de histórico...")

# Ajuste Fino: n_estimators=200 (mais árvores), max_depth=10 (evita decorar demais)
modelo = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
modelo.fit(X_treino, y_treino)

# --- 6. AVALIAÇÃO ---
previsoes = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, previsoes)

print("\n" + "=" * 40)
print(f"🏆 NOVA ACURÁCIA: {acuracia:.2%}")
print("=" * 40)

print("\nMatriz de Confusão:")
print(confusion_matrix(y_teste, previsoes))
