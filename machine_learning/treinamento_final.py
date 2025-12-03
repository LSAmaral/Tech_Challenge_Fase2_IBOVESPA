import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# --- 1. FUNÇÕES ---
def calcular_rsi(series, janela=14):
    delta = series.diff()
    ganho = (delta.where(delta > 0, 0)).rolling(window=janela).mean()
    perda = (-delta.where(delta < 0, 0)).rolling(window=janela).mean()
    rs = ganho / perda
    return 100 - (100 / (1 + rs))


# --- 2. PREPARAÇÃO ---
print("⚙️ Carregando dados...")
df = pd.read_csv("ibovespa_preparado.csv")
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values(by="data", ascending=True).reset_index(drop=True)

# --- 3. FEATURES AVANÇADAS (LAGS) ---
# Retorno simples
df["retorno"] = df["fechamento"].pct_change()

# Lags: O que aconteceu ontem? E anteontem?
# Isso ajuda o modelo a ver padrões de "sobe-desce-sobe"
df["retorno_lag1"] = df["retorno"].shift(1)
df["retorno_lag2"] = df["retorno"].shift(2)
df["retorno_lag3"] = df["retorno"].shift(3)

# Indicadores Técnicos
df["media_7"] = df["fechamento"].rolling(7).mean()
df["media_21"] = df["fechamento"].rolling(21).mean()
df["rsi"] = calcular_rsi(df["fechamento"], 14)

# Distância da Média (O preço esticou demais?)
df["distancia_media_21"] = df["fechamento"] / df["media_21"]

# --- ALVO ---
df["fechamento_amanha"] = df["fechamento"].shift(-1)
df["target"] = (df["fechamento_amanha"] > df["fechamento"]).astype(int)

# Limpeza (Os lags criam NaNs no começo)
df = df.dropna()

# --- 4. DIVISÃO ---
dias_teste = 30
indice_corte = len(df) - dias_teste

features = ["retorno_lag1", "retorno_lag2", "rsi", "distancia_media_21", "media_7"]
X = df[features]
y = df["target"]

X_treino = X.iloc[:indice_corte]
y_treino = y.iloc[:indice_corte]
X_teste = X.iloc[indice_corte:]
y_teste = y.iloc[indice_corte:]

# --- 5. TREINAMENTO COM BALANCEAMENTO ---
print(f"🤖 Treinando modelo (Gradient Boosting)...")

# Troquei para GradientBoosting (costuma ser melhor para tendências que o Random Forest)
modelo = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
modelo.fit(X_treino, y_treino)

# --- 6. AVALIAÇÃO ---
previsoes = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, previsoes)

print("\n" + "=" * 40)
print(f"🏆 ACURÁCIA FINAL: {acuracia:.2%}")
print("=" * 40)

print("Matriz de Confusão:")
print(confusion_matrix(y_teste, previsoes))
