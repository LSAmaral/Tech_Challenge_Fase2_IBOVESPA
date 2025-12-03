import os

NOME_ARQUIVO = "Dados Históricos - Ibovespa.csv"
NOME_SAIDA = "ibovespa_preparado.csv"  # <--- O Arquivo do Futuro


def converter_valor_financeiro(texto_valor):
    texto_limpo = texto_valor.replace('"', "").replace(".", "").replace(",", ".")
    try:
        return float(texto_limpo)
    except ValueError:
        return 0.0


def converter_volume(texto_vol):
    # Agora sabemos que o 'K' existe mesmo! Nossa prevenção valeu a pena.
    texto_limpo = texto_vol.replace('"', "").replace(",", ".").upper().strip()

    if not texto_limpo or texto_limpo == "NAN":
        return 0.0

    multiplicador = 1

    if "B" in texto_limpo:
        multiplicador = 1_000_000_000
        texto_limpo = texto_limpo.replace("B", "")
    elif "M" in texto_limpo:
        multiplicador = 1_000_000
        texto_limpo = texto_limpo.replace("M", "")
    elif "K" in texto_limpo:
        multiplicador = 1_000
        texto_limpo = texto_limpo.replace("K", "")

    try:
        return float(texto_limpo) * multiplicador
    except ValueError:
        return 0.0


def processar_arquivo():
    print(f"--- Processando: {NOME_ARQUIVO} ---")

    if not os.path.exists(NOME_ARQUIVO):
        print("Arquivo não encontrado!")
        return

    with open(NOME_ARQUIVO, "r", encoding="utf-8") as arquivo:
        linhas = arquivo.readlines()

    dados_limpos = []

    for linha in linhas[1:]:
        linha = linha.strip()
        partes = linha.split('","')  # O corte cirúrgico

        if len(partes) < 5:
            continue

        data_br = partes[0].replace('"', "").replace("\ufeff", "")
        preco_str = partes[1]
        vol_str = partes[5]

        # 1. Data
        try:
            dia, mes, ano = data_br.split(".")
            data_iso = f"{ano}-{mes}-{dia}"
        except ValueError:
            continue

        # 2. Valores
        preco_float = converter_valor_financeiro(preco_str)
        vol_float = converter_volume(vol_str)

        item = {"data": data_iso, "fechamento": preco_float, "volume": vol_float}
        dados_limpos.append(item)

    print(f"✅ Conversão concluída na memória! {len(dados_limpos)} registros prontos.")

    # --- ETAPA NOVA: SALVAR NO DISCO ---
    print(f"💾 Salvando em: {NOME_SAIDA}...")

    with open(NOME_SAIDA, "w", encoding="utf-8") as f_saida:
        # Escreve o cabeçalho limpo
        f_saida.write("data,fechamento,volume\n")

        # Escreve os dados linha a linha
        for item in dados_limpos:
            # f-string montando o CSV: 2025-12-02,161092.0,8430000000.0
            linha_csv = f"{item['data']},{item['fechamento']},{item['volume']}\n"
            f_saida.write(linha_csv)

    print("🚀 SUCESSO! Arquivo pronto para análise.")


if __name__ == "__main__":
    processar_arquivo()
