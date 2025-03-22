import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pmdarima.arima import ARIMA  # Importante: Importar ARIMA de pmdarima
import joblib
import os

from google.cloud import storage
from io import BytesIO
from datetime import datetime

# =========================================
# Configurações Iniciais da Página
# =========================================
st.set_page_config(
    page_title="Previsão IBOVESPA com ARIMA",
    layout="wide",  # Para permitir la columna lateral
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
# Columna de Integrantes (Sidebar)
# =========================================
with st.sidebar:
    st.subheader("Integrantes do Grupo:")
    st.markdown("""
    - Rosicleia Cavalcante Mota
    - Guillermo Jesus Camahuali Privat
    - Kelly Priscilla Matos Campos
    """)

    st.markdown("""
    Este aplicativo exibe a série histórica do IBOVESPA de 2015 a 2025,
    bem como as previsões futuras usando um modelo ARIMA treinado.
    """)

# =========================================
# Configurações do Google Cloud Storage
# =========================================
# Supondo que as credenciais estejam configuradas via Streamlit Cloud
PROJECT_ID = os.getenv("PROJECT_ID", "ibovespa-data-project")  # Reemplaza con tu Project ID
BUCKET_NAME = os.getenv("BUCKET_NAME", "ibovespa-models")  # Reemplaza con el nombre de tu bucket
BLOB_NAME = "modelos/modelo_arima.pkl"  # Arquivo do modelo ARIMA no GCS

def baixar_modelo_arima_gcs(bucket_name, blob_name):
    """Baixa o arquivo .pkl do modelo ARIMA a partir do GCS."""
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        model_bytes = blob.download_as_bytes()
        modelo_arima = joblib.load(BytesIO(model_bytes))
        st.success("Modelo ARIMA baixado com sucesso do GCS!")  # Mensaje de éxito
        return modelo_arima
    except Exception as e:
        st.error(f"Erro ao baixar modelo ARIMA do GCS: {e}")
        st.error(f"Detalhes do erro: {e}")  # Información detallada del error
        return None


@st.cache_resource  # st.cache_resource para modelos
def carregar_modelo():
    """Carrega o modelo ARIMA do GCS apenas uma vez (cacheado)."""
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
# Carregamento dos Dados de IBOVESPA
# =========================================
@st.cache_data
def carregar_dados():
    """Função simulada: Carrega dataset do IBOVESPA (exemplo)."""
    # Abaixo, você pode substituir por BigQuery ou CSV no GCS
    # Exemplo fixo:
    datas = pd.date_range("2015-01-01", "2025-03-18", freq="D")
    np.random.seed(42)
    valores = np.cumsum(np.random.normal(0, 200, len(datas))) + 80000
    df = pd.DataFrame({"Data": datas, "Fechamento": valores})
    return df

df = carregar_dados()

# =========================================
# Ignorar últimos 1 ou 2 dias se suspeitos
# =========================================
# Exemplo: ignora 1 dia
quant_dias_ignorar = 1
df = df.iloc[:-quant_dias_ignorar]  # remove as últimas N linhas

# Convertendo Data para DateTime se precisar
df["Data"] = pd.to_datetime(df["Data"])
df.sort_values("Data", inplace=True)

st.markdown("### Visualização da Série Histórica do IBOVESPA (2015–2025)")

# =========================================
# Gráfico Série Histórica
# =========================================
fig_hist = px.line(df, x="Data", y="Fechamento",
                   title="Série Histórica do IBOVESPA",
                   labels={"Data": "Data", "Fechamento": "Fechamento (pts)"},
                   width=800, height=400)  # Aumenta el ancho
st.plotly_chart(fig_hist)

st.markdown("""
**Explicação**: A linha acima mostra a evolução do índice IBOVESPA
ao longo do tempo, no período de 2015 a 2025 (dados simulados para este exemplo).
""")

# =========================================
# Predição para 60 dias
# =========================================
st.markdown("### Previsão para os Próximos 60 Dias")

# Precisamos do(s) valor(es) mais recente(s) para iniciar a predição
# ARIMA do pmdarima aceita predizer usando modelo_arima.predict(n_periods=60)
n_periods = 60

# Predizendo com base na série (exemplo simplificado)
serie_treinamento = df["Fechamento"].values

try:
    forecast, conf_int = modelo_arima.predict(n_periods=n_periods, return_conf_int=True)
except Exception as e:
    st.error(f"Falha ao prever com ARIMA: {e}")
    st.stop()

# Datas futuras
ultima_data = df["Data"].iloc[-1]
datas_futuras = pd.date_range(start=ultima_data + pd.Timedelta(days=1),
                              periods=n_periods, freq="D")

# Montar DataFrame de previsões
df_forecast = pd.DataFrame({
    "Data": datas_futuras,
    "Fechamento_Previsto": forecast,
    "Lower": conf_int[:, 0],
    "Upper": conf_int[:, 1]
})

# =========================================
# Gráfico de Previsão
# =========================================
fig_fore = go.Figure()

# Série Histórica (apenas últimos 100 dias para não poluir)
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

# Linha da Previsão
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
    width=800,  # Aumenta el ancho
    height=500
)

st.plotly_chart(fig_fore)

st.markdown("""
**Explicação**: Em vermelho, vemos a previsão para os próximos 60 dias,
e a faixa cinza representa o intervalo de confiança (inferior e superior).
""")

# =========================================
# Gráfico de Resíduos
# =========================================
st.markdown("### Análise de Resíduos do Modelo")
try:
    residuos = modelo_arima.resid() # Usar resid() para pmdarima
    fig_res = px.histogram(residuos, nbins=30, text_auto=True,
                           title="Distribuição dos Resíduos do Modelo ARIMA", width=800)  # Aumenta el ancho
    st.plotly_chart(fig_res)
except Exception as e:
    st.error(f"Erro ao analisar resíduos: {e}")

st.markdown("""
**Explicação**: O ideal é que os resíduos sejam próximos de uma distribuição normal
centrada em 0, indicando que o modelo não apresenta vieses significativos.
""")

# =========================================
# Cálculo da Métrica MAPE (Exemplo)
# =========================================
st.markdown("### Métrica de Acurácia: MAPE")

# Supondo que modelo_arima tenha sido treinado em df, ou temos um holdout
# Exemplo simples: usar a própria série para comparar
n_test = 30  # Exemplo: últimos 30 dias foram de teste (na vida real, separar dataset)
serie_treinamento = df["Fechamento"].values
treino = serie_treinamento[:-n_test]
teste = serie_treinamento[-n_test:]

# Reajustar si necesario. Aquí simplificado, sin re-treinar
try:
    #Es fundamental reentrenar el modelo con los datos de entrenamiento antes de predecir
    modelo_arima.fit(treino)
    forecast_teste, _ = modelo_arima.predict(n_periods=n_test, return_conf_int=True)

    # Cálculo do MAPE
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    erro_mape = mape(teste, forecast_teste)
    st.write(f"**MAPE**: {erro_mape:.2f}%")

    acuracia = 100 - erro_mape
    st.write(f"**Acurácia General**: {acuracia:.2f}%")

    if acuracia > 70:
        st.success("Acurácia do modelo acima de 70%!")
    else:
        st.warning("Acurácia do modelo abaixo de 70%. Considere melhorar o modelo.")


except Exception as e:
    st.warning(f"Não foi possível calcular o MAPE de forma demonstrativa: {e}")

st.markdown("""
**Explicação**: O MAPE (Mean Absolute Percentage Error) indica o erro percentual
médio entre valores reais e previstos. Quanto menor, melhor.
""")

st.success("Aplicação finalizada! Sinta-se à vontade para explorar ou ajustar parâmetros.")