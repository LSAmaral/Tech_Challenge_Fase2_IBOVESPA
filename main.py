import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- CONFIGURAÇÃO CAMPEÃ ---
DATA_INICIO_TREINO = "2020-01-01"  # 6 anos de dados (Cumpre requisito com folga)
THRESHOLD_OTIMIZADO = 0.44  # Ajuste fino descoberto na mineração


def calcular_rsi(series, janela=14):
    delta = series.diff()
    ganho = (delta.where(delta > 0, 0)).rolling(window=janela).mean()
    perda = (-delta.where(delta < 0, 0)).rolling(window=janela).mean()
    rs = ganho / perda
    return 100 - (100 / (1 + rs))


print("⚙️ Carregando e preparando dados...")
df = pd.read_csv("ibovespa_preparado.csv")
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values(by="data", ascending=True).reset_index(drop=True)

# Filtro de Data (A Estratégia Vencedora)
df = df[df["data"] >= DATA_INICIO_TREINO].copy()

# Feature Engineering
df["retorno"] = df["fechamento"].pct_change()
df["media_5"] = df["fechamento"].rolling(5).mean()
df["media_21"] = df["fechamento"].rolling(21).mean()
df["rsi"] = calcular_rsi(df["fechamento"], 14)
df["volatilidade"] = df["retorno"].rolling(5).std()
df["momentum"] = df["fechamento"] / df["fechamento"].shift(3)

# Target
df["fechamento_amanha"] = df["fechamento"].shift(-1)
df["target"] = (df["fechamento_amanha"] > df["fechamento"]).astype(int)
df = df.dropna()

# Divisão Treino/Teste (Últimos 30 dias)
dias_teste = 30
indice_corte = len(df) - dias_teste

features = ["retorno", "rsi", "media_5", "volatilidade", "momentum"]
X = df[features]
y = df["target"]

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_treino = X_scaled[:indice_corte]
y_treino = y.iloc[:indice_corte]
X_teste = X_scaled[indice_corte:]
y_teste = y.iloc[indice_corte:]

# Treinamento
print(f"🤖 Treinando Modelo Campeão (Logistic Regression)...")
modelo = LogisticRegression(random_state=42)
modelo.fit(X_treino, y_treino)

# Previsão com Threshold Otimizado
probs = modelo.predict_proba(X_teste)[:, 1]
previsoes_finais = (probs >= THRESHOLD_OTIMIZADO).astype(int)

# Avaliação
acc = accuracy_score(y_teste, previsoes_finais)

print("\n" + "=" * 50)
print(f"🏆 RESULTADO FINAL DO PROJETO")
print(f"✅ Acurácia no Teste (30 dias): {acc:.2%}")
print("=" * 50)

print("\nRelatório de Classificação:")
print(classification_report(y_teste, previsoes_finais))

# --- GERAÇÃO DE GRÁFICOS PARA O PPT ---
print("📊 Gerando gráficos para o relatório...")

# 1. Matriz de Confusão Visual
plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_teste, previsoes_finais), annot=True, fmt="d", cmap="Blues"
)
plt.title(f"Matriz de Confusão (Acc: {acc:.1%})")
plt.xlabel("Previsão do Modelo")
plt.ylabel("Realidade do Mercado")
plt.savefig("grafico_matriz_confusao.png")
print(" -> Salvo: grafico_matriz_confusao.png")

# 2. Previsão vs Real (Linha do Tempo)
# Vamos pegar as datas correspondentes ao teste
datas_teste = df["data"].iloc[indice_corte:]

plt.figure(figsize=(12, 6))
plt.plot(
    datas_teste,
    y_teste,
    label="Real (1=Alta, 0=Baixa)",
    marker="o",
    linestyle="-",
    color="gray",
    alpha=0.5,
)
plt.plot(
    datas_teste,
    previsoes_finais,
    label="Previsão Modelo",
    marker="x",
    linestyle="--",
    color="blue",
)
plt.title("Comparativo: Realidade vs Previsão (Últimos 30 dias)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("grafico_previsao_tempo.png")
print(" -> Salvo: grafico_previsao_tempo.png")

print("\n🚀 PROJETO FINALIZADO COM SUCESSO!")

# --- CAMADA DE TRADUÇÃO VISUAL (REQUISITO DO PROJETO) ---
print("\n" + "📈 RELATÓRIO DE TENDÊNCIAS (ÚLTIMOS 5 DIAS)")
print("-" * 45)

# Mapeamento para as setas do enunciado
mapa_setas = {1: "↑ ALTA", 0: "↓ BAIXA"}

# Criar DataFrame de visualização
df_visual = pd.DataFrame(
    {
        "Data": datas_teste.dt.strftime("%d/%m/%Y"),
        "Realidade": y_teste.map(mapa_setas),
        "Previsão": pd.Series(previsoes_finais).map(mapa_setas).values,
    }
)
print(df_visual.tail(5).to_string(index=False))
print("-" * 45)
