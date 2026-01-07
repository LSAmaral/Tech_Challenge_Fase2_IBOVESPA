# Tech Challenge - Fase 2 (Data Analytics)

## 🎯 Objetivo
Desenvolver um modelo preditivo capaz de prever se o índice **IBOVESPA** irá fechar em alta ou baixa no dia seguinte, servindo como ferramenta de suporte à decisão para analistas quantitativos de um fundo de investimento.

## 🏆 Resultados e Performance
* **Acurácia Final:** 76.67% (Meta mínima: 75%).
* **Modelo Utilizado:** Regressão Logística (Logistic Regression).
* **Período de Teste:** Últimos 30 dias de dados disponíveis (Conjunto isolado).

---

## 📈 Visão Gerencial (Interpretação de Resultados)
O modelo traduz probabilidades estatísticas em sinais direcionais claros ($\uparrow$ e $\downarrow$) para facilitar a leitura por parte da mesa de operações:

| Data | Fechamento Real | Tendência Real | Previsão do Modelo | Resultado |
| :--- | :--- | :---: | :---: | :---: |
| 27/01/2025 | 128.500 | ↑ | ↑ | ✅ |
| 28/01/2025 | 127.200 | ↓ | ↓ | ✅ |
| 29/01/2025 | 129.100 | ↑ | ↑ | ✅ |
| 30/01/2025 | 128.800 | ↓ | ↑ | ❌ |

---

## 🧪 Metodologia Técnica

### 1. Processamento de Dados (ETL)
* **Tratamento de Strings:** Conversão de volumes financeiros com sufixos (K, M, B) para valores numéricos (`float`).
* **Janela Temporal:** Utilização de dados históricos desde Janeiro de 2020 para garantir uma base de treino robusta.

### 2. Engenharia de Atributos (Features)
Para tratar a natureza ruidosa e sequencial do mercado financeiro, foram criadas as seguintes variáveis:
* **RSI (Relative Strength Index):** Identificação de exaustão de compra/venda.
* **Lags de Retorno:** Inclusão de retornos de dias anteriores (t-1, t-2) para fornecer memória ao modelo.
* **Médias Móveis:** Captura de tendências de curto (5 dias) e médio prazo (21 dias).

### 3. Justificativa do Modelo e Trade-offs
* **Escolha:** A **Regressão Logística** foi selecionada pela sua estabilidade e alta interpretabilidade. Permite aos analistas entenderem o peso de cada indicador na previsão.
* **Overfitting:** Optou-se por um modelo linear para evitar que o algoritmo "decore" o passado, garantindo generalização para dados futuros.
* **Threshold:** O limiar de decisão foi otimizado para **0.44**, permitindo uma melhor captura de movimentos de alta no índice.

---

## 💻 Como Rodar o Projeto

1.  **Configurar Ambiente:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
2.  **Instalar Dependências:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Executar Pipeline:**
    ```bash
    python main.py
    ```