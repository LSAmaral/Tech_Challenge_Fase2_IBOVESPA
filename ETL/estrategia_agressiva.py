import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. PREPARAÇÃO DOS DADOS ---
print("⚙️ Carregando dados...")
df = pd.read_csv("ibovespa_preparado.csv")
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values(by="data", ascending=True).reset_index(drop=True)

# ESTRATÉGIA NOVA: Cortar o passado distante!
# Vamos treinar apenas com dados a partir de 2023.
# O mercado antigo (2018-2022) só atrapalha a previsão de curto prazo agora.
data_corte_inicio = "2023-01-01"
df = df[df["data"] >= data_corte_inicio].copy()
print(f"✂️ Usando dados apenas a partir de {data_corte_inicio}")

# --- 2. FEATURES ---
df["retorno"] = df["fechamento"].pct_change()
df["media_5"] = df["fechamento"].rolling(5).mean()
df["media_21"] = df["fechamento"].rolling(21).mean()

# RSI Simplificado (14 dias)
delta = df["fechamento"].diff()
ganho = (delta.where(delta > 0, 0)).rolling(window=14).mean()
perda = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = ganho / perda
df["rsi"] = 100 - (100 / (1 + rs))

# Tendência: Preço está acima da média curta? (1 = Sim, 0 = Não)
df["tendencia_curta"] = (df["fechamento"] > df["media_5"]).astype(int)

# Target
df["fechamento_amanha"] = df["fechamento"].shift(-1)
df["target"] = (df["fechamento_amanha"] > df["fechamento"]).astype(int)

df = df.dropna()

# --- 3. DIVISÃO E ESCALA ---
dias_teste = 30
indice_corte = len(df) - dias_teste

features = ["retorno", "rsi", "tendencia_curta", "media_5", "media_21"]
X = df[features]
y = df["target"]

# Normalização (Importante para Regressão Logística e Redes Neurais)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_treino = X_scaled[:indice_corte]
y_treino = y.iloc[:indice_corte]
X_teste = X_scaled[indice_corte:]
y_teste = y.iloc[indice_corte:]

# --- 4. TREINAMENTO HÍBRIDO ---
print(f"🤖 Treinando com {len(X_treino)} dias recentes...")

# Vamos testar Regressão Logística (Modelo Linear costuma pegar melhor a tendência direta)
modelo = LogisticRegression(random_state=42)
modelo.fit(X_treino, y_treino)

# --- 5. HACK DE PROBABILIDADE ---
# Em vez de predict() direto (que corta em 50%), vamos pegar a probabilidade.
# Se o modelo der > 40% de chance de subir, a gente aposta que sobe.
# Isso corrige o pessimismo do modelo.
probabilidades = modelo.predict_proba(X_teste)[:, 1]  # Pega chance de ser "1" (Alta)
threshold = 0.45  # <--- O PULO DO GATO (Baixamos a régua)

previsoes_ajustadas = (probabilidades >= threshold).astype(int)

acuracia = accuracy_score(y_teste, previsoes_ajustadas)

print("\n" + "=" * 40)
print(f"🏆 ACURÁCIA (ESTRATÉGIA RECENTE): {acuracia:.2%}")
print("=" * 40)

print("Matriz de Confusão:")
print(confusion_matrix(y_teste, previsoes_ajustadas))

# Debug: Ver o que ele previu de verdade
# print("\nProbabilidades de Alta:", probabilidades)
