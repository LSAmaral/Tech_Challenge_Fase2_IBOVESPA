import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. PREPARAÇÃO (Repetindo a lógica das Features) ---
print("⚙️ Preparando dados...")
df = pd.read_csv("ibovespa_preparado.csv")
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values(by="data", ascending=True).reset_index(drop=True)

# Criando Features (Variáveis explicativas)
# Dica: Adicionei o 'RSI' (Índice de Força Relativa) simplificado aqui
# Se o preço subiu muito nos últimos dias, o RSI sobe. Ajuda o modelo.
df["retorno"] = df["fechamento"].pct_change()  # Quanto subiu/desceu em %
df["media_7"] = df["fechamento"].rolling(7).mean()
df["media_21"] = df["fechamento"].rolling(21).mean()
df["volatilidade"] = df["retorno"].rolling(7).std()  # Desvio padrão (medo do mercado)

# Criando Target
df["fechamento_amanha"] = df["fechamento"].shift(-1)
df["target"] = (df["fechamento_amanha"] > df["fechamento"]).astype(int)

# Limpeza de NaN
df = df.dropna()

# --- 2. DIVISÃO TREINO vs TESTE (Respeitando o Tempo) ---
# O PDF pede os últimos 30 dias para teste
dias_teste = 30
indice_corte = len(df) - dias_teste

# Definição de X (Dados para analisar) e y (Resposta correta)
features = ["fechamento", "volume", "media_7", "media_21", "volatilidade"]
X = df[features]
y = df["target"]

# A Corte Temporal (Sem embaralhar!)
X_treino = X.iloc[:indice_corte]
y_treino = y.iloc[:indice_corte]

X_teste = X.iloc[indice_corte:]
y_teste = y.iloc[indice_corte:]

print(f"📊 Dados de Treino: {len(X_treino)} dias (Passado)")
print(f"🔮 Dados de Teste:  {len(X_teste)} dias (Futuro Próximo)")

# --- 3. TREINAMENTO (A Mágica) ---
print("\n🤖 Treinando o modelo RandomForest...")
# n_estimators=100 -> Cria 100 árvores de decisão
# random_state=42 -> Garante que o resultado seja sempre o mesmo (reprodutibilidade)
modelo = RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=42)
modelo.fit(X_treino, y_treino)

# --- 4. AVALIAÇÃO ---
print("🎯 Realizando previsões...")
previsoes = modelo.predict(X_teste)

acuracia = accuracy_score(y_teste, previsoes)
print("-" * 40)
print(f"🏆 ACURÁCIA FINAL: {acuracia:.2%}")
print("-" * 40)

print("\nRelatório Detalhado:")
print(classification_report(y_teste, previsoes))

print("\nMatriz de Confusão (Acertos vs Erros):")
# [Verdadeiro Negativo, Falso Positivo]
# [Falso Negativo, Verdadeiro Positivo]
print(confusion_matrix(y_teste, previsoes))
