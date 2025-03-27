import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pmdarima.arima import ARIMA # Ou apenas 'import pmdarima' se usar auto_arima depois
import joblib
import os
import traceback # Para logs mais detalhados se necessário

# Bibliotecas para acessar GCS e BigQuery
from google.cloud import storage, bigquery # Adicionado bigquery
from google.oauth2 import service_account
from io import BytesIO
from datetime import datetime
import copy # Para copiar o modelo

# =========================================
# Configurações Iniciais da Página
# =========================================
st.set_page_config(
    page_title="Previsão IBOVESPA com ARIMA (Dados Reais)", # Título ajustado
    layout="wide",
    initial_sidebar_state="auto"
)

# =========================================
# Cabeçalho e Contexto
# =========================================
st.title("Previsão do IBOVESPA com Modelo ARIMA (Dados Reais do BigQuery)") # Título ajustado
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
    Este aplicativo carrega um modelo ARIMA pré-treinado do GCS
    e o aplica sobre os **dados históricos reais do IBOVESPA**
    carregados do Google BigQuery para visualização e previsão.
    """) # Descrição ajustada

# =========================================
# Parâmetros de Projeto e Segredos (Adicionar BQ)
# =========================================
# Lendo as variáveis de ambiente do "Secrets" do Streamlit
PROJECT_ID = st.secrets.get("PROJECT_ID", "ibovespa-data-project")
BUCKET_NAME = st.secrets.get("BUCKET_NAME", "ibovespa-models")
BLOB_NAME = "modelos/modelo_arima.pkl"
BQ_DATASET = st.secrets.get("BQ_DATASET", "ibovespa_dataset") # Adicionado
BQ_TABLE = st.secrets.get("BQ_TABLE", "dados_historicos") # Adicionado

# Verificar se as credenciais existem
if "google_credentials" not in st.secrets:
    st.error("Credenciais do Google ('google_credentials') não encontradas nos Secrets do Streamlit.")
    st.stop()

try:
    creds_dict = dict(st.secrets["google_credentials"])
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
except Exception as e:
    st.error(f"Erro ao carregar credenciais do Google a partir dos Secrets: {e}")
    st.stop()

# =========================================
# Função para Baixar o Modelo do GCS (igual, mas usa 'credentials')
# =========================================
def baixar_modelo_arima_gcs(bucket_name, blob_name, credentials):
    """Baixa o arquivo .pkl do modelo ARIMA a partir do GCS."""
    try:
        client = storage.Client(project=PROJECT_ID, credentials=credentials)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        model_bytes = blob.download_as_bytes()
        modelo_arima = joblib.load(BytesIO(model_bytes))
        st.success("Modelo ARIMA baixado com sucesso do GCS!")
        return modelo_arima
    except Exception as e:
        st.error(f"Erro ao baixar modelo ARIMA do GCS: {e}")
        return None

# =========================================
# Função para Carregar Dados Reais do BigQuery << NOVO >>
# =========================================
@st.cache_data(ttl=3600) # Cache por 1 hora para não sobrecarregar BQ
def carregar_dados_reais_bq(project_id, bq_dataset, bq_table, credentials):
    """Carrega dados históricos reais do BigQuery."""
    st.write("Carregando dados reais do BigQuery...")
    try:
        client = bigquery.Client(project=project_id, credentials=credentials)
        query = f"""
            SELECT Data, Fechamento
            FROM `{project_id}.{bq_dataset}.{bq_table}`
            WHERE Fechamento IS NOT NULL
            ORDER BY Data ASC
        """
        df = client.query(query).to_dataframe()

        if df.empty:
            st.warning("Nenhum dado encontrado no BigQuery.")
            return pd.DataFrame() # Retorna vazio para tratamento posterior

        st.success(f"Dados reais carregados do BigQuery: {df.shape[0]} linhas.")

        # Garantir tipos corretos e ordenação
        df["Data"] = pd.to_datetime(df["Data"])
        df["Fechamento"] = pd.to_numeric(df["Fechamento"])
        df.sort_values("Data", inplace=True)
        df = df.set_index('Data') # ARIMA funciona melhor com índice de data/hora
        df = df.asfreq('B') # Define frequência como dias úteis (Business days) - IMPORTANTE
                           # Se houver dias faltando, pode precisar de .fillna()
        df = df.fillna(method='ffill') # Preenche dias não úteis com valor anterior

        return df.reset_index() # Retorna Data como coluna para uso geral

    except Exception as e:
        st.error(f"Erro ao carregar dados do BigQuery: {e}")
        st.error(traceback.format_exc()) # Mostra mais detalhes do erro
        return pd.DataFrame()

# =========================================
# Carregando o Modelo ARIMA
# =========================================
modelo_arima_original = baixar_modelo_arima_gcs(BUCKET_NAME, BLOB_NAME, credentials)
if not modelo_arima_original:
    st.stop()

try:
    # Copia para poder atualizar sem modificar o original na memória
    modelo_arima = copy.deepcopy(modelo_arima_original)
    st.info(f"Modelo Carregado: {type(modelo_arima)}. Parâmetros: {modelo_arima.get_params()}")
except Exception as e:
    st.warning(f"Não foi possível fazer deepcopy do modelo ({e}), usando referência original.")
    modelo_arima = modelo_arima_original

# =========================================
# Carregando os Dados Reais do BigQuery << ALTERADO >>
# =========================================
df_real = carregar_dados_reais_bq(PROJECT_ID, BQ_DATASET, BQ_TABLE, credentials)

if df_real.empty:
    st.error("Aplicação interrompida por falta de dados reais.")
    st.stop()

# Garantir que Data é datetime após reset_index
df_real["Data"] = pd.to_datetime(df_real["Data"])
df_real.sort_values("Data", inplace=True)

st.write("### Amostra dos Dados Reais Carregados:")
st.dataframe(df_real.tail()) # Mostra as últimas linhas dos dados reais

# =========================================
# 1) Visualização da Série Histórica REAL << ALTERADO >>
# =========================================
st.markdown("### 1. Visualização da Série Histórica Real do IBOVESPA")
fig_hist = px.line(
    df_real, x="Data", y="Fechamento",
    title="Série Histórica Real do IBOVESPA (BigQuery)", # Título ajustado
    labels={"Data": "Data", "Fechamento": "Fechamento (pts)"},
    width=900, height=450 # Ajustado tamanho
)
st.plotly_chart(fig_hist, use_container_width=True) # Ocupa largura

st.markdown("""
**Explicação**: A linha acima mostra a evolução **real** do índice IBOVESPA
ao longo do tempo, com dados carregados diretamente do BigQuery.
""")

# =========================================
# Preparar dados de treino/teste REAIS para MAPE e Ajuste << ALTERADO >>
# =========================================
n_test = 30 # Dias para teste
if len(df_real) > n_test:
    # Usaremos 'Fechamento' como a série
    serie_completa_real = df_real.set_index('Data')['Fechamento'] # Série com índice de Data
    treino_real = serie_completa_real.iloc[:-n_test]
    teste_real = serie_completa_real.iloc[-n_test:]
    st.info(f"Dados divididos: {len(treino_real)} para treino, {len(teste_real)} para teste.")
else:
    st.error("Não há dados reais suficientes para criar conjunto de treino/teste.")
    treino_real, teste_real = None, None
    # st.stop() # Não parar, pode ainda prever

# =========================================
# Opcional: Atualizar modelo com dados de treino recentes antes de prever/testar
# =========================================
# Isso ajusta o modelo aos dados mais recentes, o que é bom para previsões
# mas altera ligeiramente os parâmetros do .pkl original.
# Se quiser usar o .pkl EXATAMENTE como foi treinado, comente a linha .update()
if treino_real is not None:
    try:
        st.write("Atualizando modelo com dados de treino recentes...")
        # Não use 'y=treino_real', apenas os dados.
        modelo_arima.update(treino_real.values) # Atualiza o modelo com os dados de treino reais
        st.success("Modelo atualizado com os dados de treino mais recentes.")
    except Exception as e:
        st.warning(f"Não foi possível atualizar o modelo com dados recentes: {e}. Usando modelo como carregado.")

# =========================================
# 2) Predição para os Próximos N dias (com base nos dados reais) << ALTERADO >>
# =========================================
st.markdown("### 2. Previsão ARIMA para os Próximos Dias (Modelo sobre Dados Reais)")
n_periods = st.slider("Selecione o número de dias para prever:", 30, 180, 60) # Slider

try:
    # Previsão a partir do fim da série completa real
    forecast, conf_int = modelo_arima.predict(n_periods=n_periods, return_conf_int=True)

    # Cria datas futuras a partir da ÚLTIMA data REAL
    ultima_data_real = df_real["Data"].iloc[-1]
    # Usar frequência de dias úteis 'B' para previsão
    datas_futuras = pd.date_range(start=ultima_data_real + pd.Timedelta(days=1),
                                  periods=n_periods, freq="B") # Frequência Dias Úteis

    df_forecast = pd.DataFrame({
        "Data": datas_futuras,
        "Fechamento_Previsto": forecast,
        "Lower": conf_int[:, 0],
        "Upper": conf_int[:, 1]
    })

    fig_fore = go.Figure()

    # Mostrar últimos N dias REAIS no gráfico + previsão
    ultimos_dias_reais = st.slider("Dias de histórico para exibir no gráfico:", 100, 500, 200)
    df_ultimos_reais = df_real.tail(ultimos_dias_reais)

    fig_fore.add_trace(go.Scatter(
        x=df_ultimos_reais["Data"],
        y=df_ultimos_reais["Fechamento"],
        mode="lines",
        name=f"Histórico Real ({ultimos_dias_reais} dias)", # Legenda ajustada
        line=dict(color="#636EFA")
    ))

    # Faixa de Confiança (gerada pelo modelo)
    fig_fore.add_trace(go.Scatter(
        x=pd.concat([df_forecast["Data"], df_forecast["Data"][::-1]]),
        y=pd.concat([df_forecast["Upper"], df_forecast["Lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(255,0,0,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="Intervalo de Confiança (Modelo)"
    ))

    # Linha de Previsão (gerada pelo modelo)
    fig_fore.add_trace(go.Scatter(
        x=df_forecast["Data"],
        y=df_forecast["Fechamento_Previsto"],
        mode="lines",
        name=f"Previsão ARIMA ({n_periods} dias)", # Legenda ajustada
        line=dict(color="red")
    ))

    fig_fore.update_layout(
        title="Previsão ARIMA (Modelo Carregado) sobre Dados Reais", # Título ajustado
        xaxis_title="Data",
        yaxis_title="Fechamento (pts)",
        height=500 # Ajustado altura
    )
    st.plotly_chart(fig_fore, use_container_width=True) # Ocupa largura

    st.markdown(f"""
    **Explicação**: A linha azul mostra os últimos {ultimos_dias_reais} dias do histórico **real** do IBOVESPA (do BigQuery).
    A linha vermelha é a **previsão** gerada pelo modelo ARIMA carregado para os próximos {n_periods} dias úteis, a partir do último ponto real.
    A faixa vermelha clara representa o **intervalo de confiança** dessa previsão, também calculado pelo modelo.
    """) # Explicação ajustada

    st.write("### Dados da Previsão:")
    st.dataframe(df_forecast.style.format({"Fechamento_Previsto": "{:.2f}", "Lower": "{:.2f}", "Upper": "{:.2f}"}))


except Exception as e:
    st.error(f"Falha ao gerar previsão ARIMA: {e}")
    st.error(traceback.format_exc())

# =========================================
# 3) Análise de Resíduos do Modelo (após ajuste/update aos dados reais) << ALTERADO >>
# =========================================
st.markdown("### 3. Análise de Resíduos do Modelo (no Ajuste aos Dados Reais)")

try:
    # Se o modelo foi atualizado com treino_real, os resíduos são dessa atualização.
    # Se não foi atualizado, .resid() pode dar erro ou retornar resíduos do treino original (se armazenado).
    # É mais significativo se o modelo foi atualizado.
    residuos = modelo_arima.resid() # Pega resíduos do último fit/update
    fig_res = px.histogram(
        residuos, nbins=40, # Aumentado bins
        title="Distribuição dos Resíduos (Ajuste/Update em Dados Reais)", # Título ajustado
        width=800
    )
    st.plotly_chart(fig_res, use_container_width=True)

    # Teste de Normalidade (Shapiro-Wilk) - Opcional
    from scipy.stats import shapiro
    if len(residuos) > 3: # Shapiro requer pelo menos 3 amostras
        stat, p_value = shapiro(residuos)
        st.write(f"**Teste de Normalidade Shapiro-Wilk:** Estatística={stat:.3f}, p-valor={p_value:.3f}")
        if p_value > 0.05:
            st.success("O p-valor > 0.05 sugere que os resíduos não são significativamente diferentes de uma distribuição normal.")
        else:
            st.warning("O p-valor <= 0.05 sugere que os resíduos podem não seguir uma distribuição normal.")

    st.markdown("""
    **Explicação**: O histograma mostra a distribuição dos erros (resíduos) do modelo após ser ajustado/atualizado com os dados de treino **reais**.
    Idealmente, os resíduos devem parecer normalmente distribuídos (formato de sino) em torno de zero. O teste de Shapiro-Wilk ajuda a verificar isso estatisticamente.
    """) # Explicação ajustada
except Exception as e:
    st.warning(f"Não foi possível gerar análise de resíduos: {e}. (Talvez o modelo não foi atualizado?)")

# =========================================
# 4) Cálculo da Métrica MAPE (Modelo vs Dados de Teste REAIS) << ALTERADO >>
# =========================================
st.markdown("### 4. Métrica de Acurácia Real: MAPE (Modelo vs Dados de Teste Reais)")

if teste_real is not None:
    try:
        # Previsão para o período de teste real
        # Usa o modelo já carregado (e possivelmente atualizado com treino_real)
        st.write(f"Gerando previsões para o período de teste ({len(teste_real)} dias)...")
        forecast_teste, _ = modelo_arima.predict(n_periods=len(teste_real), return_conf_int=True)

        # Criar DataFrame para comparar
        df_comp = pd.DataFrame({
            'Real': teste_real.values,
            'Previsto': forecast_teste
        }, index=teste_real.index)

        st.write("### Comparação Real vs. Previsto (Teste):")
        st.dataframe(df_comp.style.format("{:.2f}"))

        # Calcular MAPE
        def mape(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            non_zero_mask = y_true != 0
            return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

        erro_mape = mape(df_comp['Real'], df_comp['Previsto'])
        st.write(f"**MAPE (no conjunto de teste REAL)**: {erro_mape:.2f}%") # Texto ajustado

        acuracia = 100 - erro_mape
        st.write(f"**Acurácia (no conjunto de teste REAL)**: {acuracia:.2f}%") # Texto ajustado

        if acuracia > 70:
            st.success("Acurácia do modelo (nos dados REAIS) acima de 70%!")
        else:
            st.warning("Acurácia do modelo (nos dados REAIS) abaixo de 70%.")

        # Plotar Real vs Previsto no Teste
        fig_teste = go.Figure()
        fig_teste.add_trace(go.Scatter(x=df_comp.index, y=df_comp['Real'], mode='lines', name='Real (Teste)'))
        fig_teste.add_trace(go.Scatter(x=df_comp.index, y=df_comp['Previsto'], mode='lines', name='Previsto (Modelo)', line=dict(color='red')))
        fig_teste.update_layout(title='Comparação Real vs. Previsto no Conjunto de Teste', xaxis_title='Data', yaxis_title='Fechamento (pts)')
        st.plotly_chart(fig_teste, use_container_width=True)


    except Exception as e:
        st.error(f"Não foi possível calcular o MAPE ou comparar no teste real: {e}")
        st.error(traceback.format_exc())
else:
    st.warning("Não foi possível calcular o MAPE pois não há dados de teste definidos.")


st.markdown("""
**Explicação**: O MAPE aqui mede o erro percentual médio do modelo ARIMA carregado ao prever o conjunto de **teste REAL**, após ter sido opcionalmente atualizado com os dados de **treino REAIS**. A acurácia (100% - MAPE) indica o quão bem o modelo generalizou para dados **reais** que não viu durante o ajuste/atualização recente.
""") # Explicação ajustada

st.success("Aplicação finalizada!")
