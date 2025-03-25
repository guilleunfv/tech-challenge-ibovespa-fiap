import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pmdarima.arima import ARIMA  # Importante: Importar ARIMA de pmdarima
import joblib
import os

# Bibliotecas para acessar o GCS usando credenciais do Secrets
from google.cloud import storage
from google.oauth2 import service_account
from io import BytesIO
from datetime import datetime

# =========================================
# Configurações Iniciais da Página
# =========================================
st.set_page_config(
    page_title="Previsão IBOVESPA com ARIMA",
    layout="wide",  # layout "wide" para expandir
    initial_sidebar_state="auto"
)

# =========================================
# Cabeçalho e Contexto
# =========================================
st.title("Previsão do IBOVESPA com Modelo ARIMA")
st.markdown("""
**Tech Challenge** - FIAP  
""")

# =========================================
# Coluna de Integrantes (Sidebar)
# =========================================
with st.sidebar:
    st.subheader("Integrantes do Grupo:")
    st.markdown("""
    - Guillermo Jesus Camahuali Privat  
    - Kelly Priscilla Matos Campos
    """)

    st.markdown("""
    Este aplicativo exibe a série histórica do IBOVESPA de 2015 a 2025,
    bem como as previsões futuras usando um modelo ARIMA treinado.
    """)

# =========================================
# Parâmetros de Projeto e Segredos
# =========================================
# Lendo as variáveis de ambiente do "Secrets" do Streamlit
PROJECT_ID = st.secrets.get("PROJECT_ID", "ibovespa-data-project")
BUCKET_NAME = st.secrets.get("BUCKET_NAME", "ibovespa-models")
BLOB_NAME = "modelos/modelo_arima.pkl"  # Arquivo do modelo ARIMA no GCS

# =========================================
# Função para Baixar o Modelo do GCS
# =========================================
def baixar_modelo_arima_gcs(bucket_name, blob_name):
    """Baixa o arquivo .pkl do modelo ARIMA a partir do GCS usando credenciais do Secrets."""
    try:
        # Carrega as credenciais do secrets (formato service_account)
        creds_dict = dict(st.secrets["google_credentials"])
        credentials = service_account.Credentials.from_service_account_info(creds_dict)

        # Cria o cliente do Cloud Storage com as credenciais
        client = storage.Client(project=PROJECT_ID, credentials=credentials)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Faz download do conteúdo em bytes e carrega o modelo
        model_bytes = blob.download_as_bytes()
        modelo_arima = joblib.load(BytesIO(model_bytes))

        st.success("Modelo ARIMA baixado com sucesso do GCS!")
        return modelo_arima

    except Exception as e:
        st.error(f"Erro ao baixar modelo ARIMA do GCS: {e}")
        return None

# =========================================
# Cache do Modelo
# =========================================
@st.cache_resource  # Para evitar recarregar o modelo em cada run
def carregar_modelo():
    st.write("Carregando modelo ARIMA do GCS...")
    modelo = baixar_modelo_arima_gcs(BUCKET_NAME, BLOB_NAME)
    return modelo

# =========================================
# Carregando o Modelo ARIMA
# =========================================
modelo_arima = carregar_modelo()
if not modelo_arima:
    st.stop()

# =========================================
# Carregando os Dados 
# =========================================
@st.cache_data
def carregar_dados():
    """Função simulada que gera um DataFrame com datas de 2015-01-01 até 2025-03-18."""
    datas = pd.date_range("2015-01-01", "2025-03-18", freq="D")
    np.random.seed(42)
    valores = np.cumsum(np.random.normal(0, 200, len(datas))) + 80000
    df = pd.DataFrame({"Data": datas, "Fechamento": valores})
    return df

df = carregar_dados()

# Ignorar últimos 1 dia (exemplo) caso suspeito
quant_dias_ignorar = 1
df = df.iloc[:-quant_dias_ignorar]

df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
df.sort_values("Data", inplace=True)

# =========================================
# 1) Visualização da Série Histórica
# =========================================
st.markdown("### Visualização da Série Histórica do IBOVESPA (2015–2025)")
fig_hist = px.line(
    df, x="Data", y="Fechamento",
    title="Série Histórica do IBOVESPA",
    labels={"Data": "Data", "Fechamento": "Fechamento (pts)"},
    width=800, height=400
)
st.plotly_chart(fig_hist)

st.markdown("""
**Explicação**: A linha acima mostra a evolução do índice IBOVESPA
ao longo do tempo, no período de 2015 a 2025.
""")

# =========================================
# 2) Predição para os Próximos 60 dias
# =========================================
st.markdown("### Previsão para os Próximos 60 Dias")
n_periods = 60
serie_treinamento = df["Fechamento"].values

try:
    # Previsão simples, assumindo que o modelo já esteja treinado
    forecast, conf_int = modelo_arima.predict(n_periods=n_periods, return_conf_int=True)
except Exception as e:
    st.error(f"Falha ao prever com ARIMA: {e}")
    st.stop()

ultima_data = df["Data"].iloc[-1]
datas_futuras = pd.date_range(start=ultima_data + pd.Timedelta(days=1),
                              periods=n_periods, freq="D")

df_forecast = pd.DataFrame({
    "Data": datas_futuras,
    "Fechamento_Previsto": forecast,
    "Lower": conf_int[:, 0],
    "Upper": conf_int[:, 1]
})

fig_fore = go.Figure()

# Mostrar só últimos 100 dias no gráfico + previsão
df_ultimos = df.tail(100)

fig_fore.add_trace(go.Scatter(
    x=df_ultimos["Data"],
    y=df_ultimos["Fechamento"],
    mode="lines",
    name="Histórico (últimos 100 dias)",
    line=dict(color="#636EFA")
))

# Faixa de Confiança
fig_fore.add_trace(go.Scatter(
    x=pd.concat([df_forecast["Data"], df_forecast["Data"][::-1]]),
    y=pd.concat([df_forecast["Upper"], df_forecast["Lower"][::-1]]),
    fill="toself",
    fillcolor="rgba(99,110,250,0.2)",
    line=dict(color="rgba(255,255,255,0)"),
    hoverinfo="skip",
    name="Intervalo de Confiança"
))

# Linha de Previsão
fig_fore.add_trace(go.Scatter(
    x=df_forecast["Data"],
    y=df_forecast["Fechamento_Previsto"],
    mode="lines",
    name="Previsão ARIMA (próximos 60 dias)",
    line=dict(color="red")
))

fig_fore.update_layout(
    title="Previsão ARIMA e Intervalo de Confiança",
    xaxis_title="Data",
    yaxis_title="Fechamento (pts)",
    width=800,
    height=500
)
st.plotly_chart(fig_fore)

st.markdown("""
**Explicação**: Em vermelho, vemos a previsão para os próximos 60 dias,
e a faixa cinza representa o intervalo de confiança (inferior e superior).
""")

# =========================================
# 3) Análise de Resíduos
# =========================================
st.markdown("### Análise de Resíduos do Modelo")

try:
    # Para pmdarima, podemos usar .resid() para extrair resíduos
    residuos = modelo_arima.resid()
    fig_res = px.histogram(
        residuos, nbins=30, text_auto=True,
        title="Distribuição dos Resíduos do Modelo ARIMA",
        width=800
    )
    st.plotly_chart(fig_res)

    st.markdown("""
    **Explicação**: O ideal é que os resíduos sejam próximos de uma distribuição normal
    centrada em 0, indicando que o modelo não apresenta vieses significativos.
    """)
except Exception as e:
    st.error(f"Erro ao analisar resíduos: {e}")

# =========================================
# 4) Cálculo da Métrica MAPE
# =========================================
st.markdown("### Métrica de Acurácia: MAPE")

# Exemplo simples: separar últimos 30 dias como "teste"
n_test = 30
treino = serie_treinamento[:-n_test]
teste = serie_treinamento[-n_test:]

try:
    # Re-treinando o modelo ARIMA somente com dados de treino (exemplo)
    modelo_arima.fit(treino)
    forecast_teste, _ = modelo_arima.predict(n_periods=n_test, return_conf_int=True)

    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    erro_mape = mape(teste, forecast_teste)
    st.write(f"**MAPE**: {erro_mape:.2f}%")

    acuracia = 100 - erro_mape
    st.write(f"**Acurácia Geral**: {acuracia:.2f}%")

    if acuracia > 70:
        st.success("Acurácia do modelo acima de 70%!")
    else:
        st.warning("Acurácia do modelo abaixo de 70%. Considere melhorar o modelo.")

except Exception as e:
    st.warning(f"Não foi possível calcular o MAPE de forma demonstrativa: {e}")

st.markdown("""
**Explicação**: O MAPE (Mean Absolute Percentage Error) indica o erro percentual
médio entre valores reais e previstos. Quanto menor, melhor. A acurácia é 100% - MAPE.
""")

st.success("Aplicação finalizada! Sinta-se à vontade para explorar ou ajustar parâmetros.")
