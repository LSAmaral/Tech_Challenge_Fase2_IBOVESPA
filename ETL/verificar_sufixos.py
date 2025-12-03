import os

NOME_ARQUIVO = "Dados Históricos - Ibovespa.csv"


def descobrir_sufixos():
    print(f"--- Varrendo arquivo em busca de sufixos na coluna Volume ---")

    with open(NOME_ARQUIVO, "r", encoding="utf-8") as arquivo:
        linhas = arquivo.readlines()

    # Usamos um SET (Conjunto) porque ele não aceita repetição.
    # Se aparecer "B" 1000 vezes, o set só guarda um "B".
    sufixos_encontrados = set()

    # Pula o cabeçalho
    for linha in linhas[1:]:
        linha = linha.strip()
        partes = linha.split('","')

        if len(partes) < 6:
            continue

        # Pega a coluna de Volume (ex: "8,43B")
        vol_str = partes[5].replace('"', "").upper().strip()

        # Lógica: Pega o último caractere se ele for letra
        if vol_str and vol_str[-1].isalpha():
            sufixos_encontrados.add(vol_str[-1])
        elif vol_str == "-":
            sufixos_encontrados.add("(Traço/Sem Volume)")

    print(f"\n🔍 Sufixos encontrados em todo o arquivo: {sufixos_encontrados}")


if __name__ == "__main__":
    descobrir_sufixos()
