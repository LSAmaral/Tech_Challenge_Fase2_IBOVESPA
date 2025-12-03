import os

NOME_ARQUIVO = "Dados Históricos - Ibovespa.csv"

print(f"--- Iniciando Inspeção de: {NOME_ARQUIVO} ---\n")

if not os.path.exists(NOME_ARQUIVO):
    print(f"❌ ERRO CRÍTICO: O arquivo '{NOME_ARQUIVO}' não está na pasta.")
    print(f"Pasta atual do Python: {os.getcwd()}")
else:
    print("✅ Arquivo encontrado! Lendo as primeiras linhas...\n")

    with open(NOME_ARQUIVO, "r", encoding="utf-8") as arquivo:
        linhas = arquivo.readlines()

        print(f"📊 Total de linhas: {len(linhas)}")
        print("-" * 50)

        for i, linha in enumerate(linhas[:5]):
            print(f"Linha {i}: {repr(linha.strip())}")

    print("-" * 50)
    print("Aguardando análise...")
