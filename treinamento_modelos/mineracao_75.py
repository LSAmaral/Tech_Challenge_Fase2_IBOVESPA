import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# --- 1. CARGA E FUNÇ ÕES ---
def calcular_rsi(series, janela=14):
    delta = series.diff()
    ganho = (delta.where(delta > 0, 0)).rolling(window=janela).mean()
    perda = (-delta.where(delta < 0, 0)).rolling(window=janela).mean()
    rs = ganho / perda
    return 100 - (100 / (1 + rs))


print("⚙️ Carregando dados...")
df_full = pd.read_csv("ibovespa_preparado.csv")
df_full["data"] = pd.to_datetime(df_full["data"])
df_full = df_full.sort_values(by="data", ascending=True).reset_index(drop=True)

# Criar Features no dataset inteiro
df_full["retorno"] = df_full["fechamento"].pct_change()
df_full["media_5"] = df_full["fechamento"].rolling(5).mean()
df_full["media_21"] = df_full["fechamento"].rolling(21).mean()
df_full["rsi"] = calcular_rsi(df_full["fechamento"], 14)
df_full["volatilidade"] = df_full["retorno"].rolling(5).std()
df_full["momentum"] = df_full["fechamento"] / df_full["fechamento"].shift(3)

# Target
df_full["fechamento_amanha"] = df_full["fechamento"].shift(-1)
df_full["target"] = (df_full["fechamento_amanha"] > df_full["fechamento"]).astype(int)
df_full = df_full.dropna()

# --- 2. O LABORATÓRIO DE TESTES ---
datas_corte = ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"]
modelos = {
    "LogisticRegression": LogisticRegression(random_state=42),
    "RandomForest (Pequena)": RandomForestClassifier(
        n_estimators=50, max_depth=3, random_state=42
    ),
    "RandomForest (Grande)": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, random_state=42
    ),
    "KNN (Vizinhos)": KNeighborsClassifier(n_neighbors=5),
    "SVM (Vetores)": SVC(probability=True, random_state=42),
}

melhor_acuracia = 0
melhor_config = {}

print("\n🚀 INICIANDO MINERAÇÃO DE MODELOS...\n")

for data_inicio in datas_corte:
    # Filtra dados pelo tempo
    df = df_full[df_full["data"] >= data_inicio].copy()

    # Separa Treino/Teste
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

    for nome_modelo, modelo in modelos.items():
        # Treina
        modelo.fit(X_treino, y_treino)

        # Testa PROBABILIDADES para ajustar o Threshold
        try:
            probs = modelo.predict_proba(X_teste)[:, 1]

            # Testa limiares de 0.40 a 0.60
            for threshold in np.arange(0.4, 0.61, 0.02):
                preds = (probs >= threshold).astype(int)
                acc = accuracy_score(y_teste, preds)

                if acc > melhor_acuracia:
                    melhor_acuracia = acc
                    melhor_config = {
                        "Data Início": data_inicio,
                        "Modelo": nome_modelo,
                        "Threshold": threshold,
                        "Acurácia": acc,
                    }
                    print(
                        f"🔥 NOVO RECORDE: {acc:.2%} | {nome_modelo} | Desde {data_inicio} | Corte {threshold:.2f}"
                    )

        except:
            continue

print("\n" + "=" * 50)
print(f"🏆 MELHOR CONFIGURAÇÃO ENCONTRADA: {melhor_acuracia:.2%}")
print("=" * 50)
print(melhor_config)
