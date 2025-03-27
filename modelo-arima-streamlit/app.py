# =========================================
# Importações Necessárias
# =========================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pmdarima # Importa pmdarima (auto_arima e ARIMA estão dentro)
import joblib
import os
import traceback # Mantido para logs detalhados em caso de erro inesperado
import copy
from io import BytesIO
from datetime import datetime

# Bibliotecas Google Cloud
from google.cloud import storage, bigquery
from google.oauth2 import service_account

# Estatísticas (para teste de Shapiro-Wilk)
from scipy.stats import shapiro

# =========================================
# Configurações Iniciais da Página
# =========================================
st.set_page_config(
    page_title="Previsão IBOVESPA com ARIMA (Dados Reais)",
    layout="wide",
    initial_sidebar_state="expanded" # Sidebar aberta por padrão
)

# =========================================
# Cabeçalho e Contexto
# =========================================
st.title("IBOVESPA: Análise e Previsão com ARIMA")
st.markdown("*(Modelo pré-treinado do GCS + Dados reais do BigQuery)*")
st.markdown("---")
st.markdown("**Tech Challenge** - FIAP")


# =========================================
# Coluna de Informações (Sidebar)
# =========================================
with st.sidebar:
    st.subheader("Integrantes do Grupo:")
    st.markdown("""
    - Guillermo Jesus Camahuali Privat
    - Kelly Priscilla Matos Campos
    """)
    st.markdown("---")
    st.subheader("Sobre a Aplicação")
    st.markdown("""
    Esta aplicação demonstra o uso de um modelo ARIMA para o IBOVESPA:
    1.  Carrega um **modelo ARIMA pré-treinado** de um arquivo `.pkl` no Google Cloud Storage (GCS).
    2.  Busca os **dados históricos reais** do IBOVESPA diretamente do Google BigQuery.
    3.  Visualiza a série histórica real.
    4.  **Atualiza** o modelo com os dados mais recentes (opcionalmente).
    5.  Gera **previsões** futuras com intervalos de confiança.
    6.  Analisa os **resíduos** do modelo.
    7.  Calcula a **acurácia (MAPE)** em um conjunto de teste real.
    """)
    st.markdown("---")
    st.info("Use os sliders na área principal para ajustar os períodos de previsão e visualização.")

# =========================================
# Leitura dos Segredos do Streamlit
# =========================================
# Verifica a existência das credenciais primeiro
if "google_credentials" not in st.secrets:
    st.error("Erro Crítico: Credenciais do Google ('google_credentials') não configuradas nos Secrets do Streamlit.")
    st.stop()

# Lê os parâmetros do projeto e nomes de recursos dos secrets
try:
    PROJECT_ID = st.secrets["PROJECT_ID"]
    BUCKET_NAME = st.secrets["BUCKET_NAME"]
    BQ_DATASET = st.secrets["BQ_DATASET"]
    BQ_TABLE = st.secrets["BQ_TABLE"]
    BLOB_NAME = "modelos/modelo_arima.pkl" # Caminho fixo do modelo no GCS
except KeyError as e:
    st.error(f"Erro Crítico: Parâmetro '{e}' não encontrado nos Secrets do Streamlit. Verifique a configuração.")
    st.stop()

# Exibe configurações carregadas na sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Configurações Carregadas:")
st.sidebar.markdown(f"""
- **Projeto Google:** `{PROJECT_ID}`
- **Bucket GCS:** `{BUCKET_NAME}`
- **Modelo GCS:** `{BLOB_NAME}`
- **Dataset BigQuery:** `{BQ_DATASET}`
- **Tabela BigQuery:** `{BQ_TABLE}`
""")

# =========================================
# Função para Carregar Credenciais (Helper)
# =========================================
def carregar_credenciais_google():
    """Carrega as credenciais da conta de serviço a partir dos secrets."""
    try:
        creds_dict = dict(st.secrets["google_credentials"])
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        # st.info(f"Credenciais carregadas para: {credentials.service_account_email}. Válidas: {credentials.valid}") # Log de debug removido
        return credentials
    except Exception as e:
        st.error(f"Erro ao processar 'google_credentials' dos Secrets: {e}")
        return None

# =========================================
# Função para Baixar o Modelo do GCS (Cache de Recurso)
# =========================================
@st.cache_resource(show_spinner="Carregando modelo ARIMA do GCS...")
def baixar_modelo_arima_gcs(bucket_name, blob_name, project_id):
    """Baixa e carrega o arquivo .pkl do modelo ARIMA a partir do GCS."""
    credentials = carregar_credenciais_google()
    if credentials is None:
        st.error("Falha ao obter credenciais para acessar GCS.")
        return None

    try:
        storage_client = storage.Client(project=project_id, credentials=credentials)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
             st.error(f"Erro: Modelo não encontrado no GCS em gs://{bucket_name}/{blob_name}")
             return None

        # st.info(f"Baixando modelo de gs://{bucket_name}/{blob_name}...") # Log de debug removido
        model_bytes = blob.download_as_bytes()
        modelo_arima = joblib.load(BytesIO(model_bytes))
        st.success("Modelo ARIMA carregado com sucesso do GCS!")
        return modelo_arima
    except Exception as e:
        st.error(f"Erro ao baixar/carregar modelo ARIMA do GCS: {e}")
        st.error(traceback.format_exc()) # Mantém traceback para erros inesperados
        return None

# =========================================
# Função para Carregar Dados Reais do BigQuery (Cache de Dados)
# =========================================
@st.cache_data(ttl=3600, show_spinner="Carregando dados reais do BigQuery...") # Cache por 1 hora
def carregar_dados_reais_bq(project_id, bq_dataset, bq_table):
    """Carrega e processa inicialmente os dados históricos reais (Data, Fechamento) do BigQuery."""
    credentials_local = carregar_credenciais_google()
    if credentials_local is None:
        st.error("Falha ao obter credenciais para acessar BigQuery.")
        return None

    try:
        # Usar project_id explícito é mais claro
        bq_client = bigquery.Client(project=project_id, credentials=credentials_local)
        # st.info(f"Cliente BigQuery criado para projeto: {bq_client.project}") # Log de debug removido

        # Nomes das colunas conforme confirmado no schema
        nome_coluna_data = "Data"
        nome_coluna_fechamento = "Fechamento"
        table_id_completo = f"`{bq_client.project}.{bq_dataset}.{bq_table}`"

        # Query SQL para buscar os dados necessários
        query = f"""
            SELECT {nome_coluna_data}, {nome_coluna_fechamento}
            FROM {table_id_completo}
            WHERE {nome_coluna_fechamento} IS NOT NULL AND DATE({nome_coluna_data}) <= CURRENT_DATE()
            ORDER BY {nome_coluna_data} ASC
        """
        # st.info(f"Executando query na tabela '{bq_table}': {query}") # Log de debug removido

        df = bq_client.query(query).to_dataframe() # Requer db-dtypes instalado

        if df.empty:
            st.warning(f"A query para {table_id_completo} não retornou dados.")
            return pd.DataFrame()

        st.success(f"Dados reais carregados do BigQuery: {df.shape[0]} linhas encontradas.")

        # Renomeia colunas para o padrão esperado pelo restante do código, se necessário
        # (Neste caso, os nomes já são 'Data' e 'Fechamento', mas deixamos por robustez)
        df.rename(columns={nome_coluna_data: 'Data', nome_coluna_fechamento: 'Fechamento'}, inplace=True)

        # Garante tipos corretos e remove duplicatas
        df["Data"] = pd.to_datetime(df["Data"])
        df["Fechamento"] = pd.to_numeric(df["Fechamento"])
        df.sort_values("Data", inplace=True)
        df = df.drop_duplicates(subset='Data', keep='last')

        return df

    except Exception as e:
        st.error(f"Erro durante a consulta ou processamento inicial do BigQuery: {e}")
        st.error(traceback.format_exc()) # Mantém traceback para erros inesperados
        return None

# =========================================
# --- Início da Lógica Principal do App ---
# =========================================

# 1. Carregar Modelo
# ------------------
modelo_arima_original = baixar_modelo_arima_gcs(BUCKET_NAME, BLOB_NAME, PROJECT_ID)

if modelo_arima_original is None:
    st.error("Aplicação interrompida: Não foi possível carregar o modelo ARIMA.")
    st.stop()

# Copia o modelo para permitir atualizações sem afetar o cache do recurso original
try:
    modelo_arima = copy.deepcopy(modelo_arima_original)
    st.info(f"Modelo ARIMA '{type(modelo_arima).__name__}' carregado. Ordem={modelo_arima.order}, Sazonal={modelo_arima.seasonal_order}")
except Exception as e:
    st.warning(f"Não foi possível fazer deepcopy do modelo ({e}). Usando referência original (pode afetar cache se 'update' for usado).")
    modelo_arima = modelo_arima_original


# 2. Carregar Dados Reais
# -----------------------
df_real_raw = carregar_dados_reais_bq(PROJECT_ID, BQ_DATASET, BQ_TABLE)

if df_real_raw is None or df_real_raw.empty:
    st.error("Aplicação interrompida: Não foi possível carregar os dados reais do BigQuery.")
    st.stop()


# 3. Pré-processar Dados (Frequência, Preenchimento)
# -------------------------------------------------
st.markdown("---")
st.subheader("3. Processamento e Visualização dos Dados Reais")
with st.spinner("Ajustando frequência e preenchendo dados..."):
    df_real = df_real_raw.copy()
    try:
        df_real = df_real.set_index('Data')
        if not df_real.index.is_monotonic_increasing:
            # st.warning("Índice de datas não monotônico. Reordenando...") # Log removido
            df_real = df_real.sort_index()

        df_real = df_real.asfreq('B') # 'B' para dias úteis
        n_preenchidos = df_real['Fechamento'].isnull().sum()
        if n_preenchidos > 0:
            # st.info(f"Preenchendo {n_preenchidos} valores ausentes (fins de semana/feriados) com 'ffill'...") # Log removido
            df_real = df_real.fillna(method='ffill')
            if df_real['Fechamento'].isnull().any():
                 # st.warning("Ainda NaN após 'ffill'. Preenchendo com 'bfill'...") # Log removido
                 df_real = df_real.fillna(method='bfill')
        df_real = df_real.reset_index()
        # st.success("Frequência ajustada e dados preenchidos.") # Log removido
    except ValueError as e:
        st.error(f"Erro ao ajustar a frequência dos dados: {e}. Usando dados originais ordenados.")
        df_real = df_real_raw.sort_values("Data").copy()

    # Garante tipo datetime final
    df_real["Data"] = pd.to_datetime(df_real["Data"])

st.success("Dados processados com sucesso.")
st.write(f"**Período dos dados:** {df_real['Data'].min().strftime('%Y-%m-%d')} a {df_real['Data'].max().strftime('%Y-%m-%d')}")
with st.expander("Visualizar últimos 5 registros processados"):
    st.dataframe(df_real.tail(), use_container_width=True)

# Gráfico Histórico Real
# st.write("**Gráfico da Série Histórica Completa:**") # Título opcional, já tem o do gráfico
fig_hist = px.line(
    df_real, x="Data", y="Fechamento",
    title="Série Histórica Real do IBOVESPA (Processada)",
    labels={"Data": "Data", "Fechamento": "Fechamento (Pontos)"},
)
fig_hist.update_layout(height=450)
st.plotly_chart(fig_hist, use_container_width=True)
st.caption("*(Gráfico interativo: use o mouse para dar zoom ou selecionar períodos)*")


# 4. Dividir Dados para Treino/Teste e Atualizar Modelo
# ----------------------------------------------------
st.markdown("---")
st.subheader("4. Preparação para Previsão e Avaliação")

n_test = 30 # Número de dias recentes reservados para teste
serie_completa_real = df_real.set_index('Data')['Fechamento'] # Série com índice de Data/Hora

if len(serie_completa_real) > n_test:
    treino_real = serie_completa_real.iloc[:-n_test]
    teste_real = serie_completa_real.iloc[-n_test:]
    st.write(f"Dados divididos: **{len(treino_real)}** obs. para ajuste/atualização do modelo, **{len(teste_real)}** obs. para teste.")

    # Opção para atualizar (reajustar) o modelo com os dados de treino mais recentes
    atualizar = st.checkbox("Atualizar modelo com dados de treino recentes?", value=True,
                            help="Reajusta o modelo carregado aos dados mais recentes disponíveis (exceto teste). Recomendado para previsões mais aderentes à dinâmica atual.")
    if atualizar:
        with st.spinner("Atualizando modelo ARIMA com dados recentes..."):
            try:
                modelo_arima.update(treino_real.values, maxiter=50) # Atualiza in-place
                st.success("Modelo ARIMA atualizado com os dados de treino.")
                # st.info(f"Parâmetros após update: {modelo_arima.get_params()}") # Log opcional removido
            except Exception as e:
                st.warning(f"Falha ao atualizar o modelo: {e}. A previsão usará o modelo sem atualização recente.")
                # st.warning(traceback.format_exc()) # Removido para não poluir usuário final
    else:
        st.info("Modelo não será atualizado. A previsão usará o modelo conforme carregado.")

else:
    st.warning("Dados insuficientes para criar conjunto de teste. O modelo não será avaliado e a previsão usará todos os dados disponíveis.")
    treino_real, teste_real = None, None # Garante que teste_real é None


# 5. Gerar e Visualizar Previsões
# -------------------------------
st.markdown("---")
st.subheader("5. Previsão Futura com ARIMA")

n_periods = st.slider("Selecione o número de dias úteis para prever:", min_value=10, max_value=180, value=60, step=10,
                      help="Número de dias úteis (seg-sex) a serem previstos a partir do último dado disponível.")

try:
    with st.spinner(f"Gerando previsão ARIMA para os próximos {n_periods} dias úteis..."):
        # Previsão a partir do fim da série completa real
        forecast, conf_int = modelo_arima.predict(n_periods=n_periods, return_conf_int=True)

        ultima_data_real = serie_completa_real.index[-1]
        datas_futuras = pd.date_range(start=ultima_data_real + pd.Timedelta(days=1),
                                      periods=n_periods, freq='B') # Dias Úteis

        df_forecast = pd.DataFrame({
            "Data": datas_futuras,
            "Fechamento_Previsto": forecast,
            "Limite_Inferior_IC95": conf_int[:, 0], # Nome mais descritivo
            "Limite_Superior_IC95": conf_int[:, 1]  # Nome mais descritivo
        })

    # Gráfico de Previsão
    fig_fore = go.Figure()
    ultimos_dias_reais = st.slider("Dias de histórico real para exibir no gráfico:", 50, 500, 200, step=50, key="hist_slider_pred")
    serie_historico_recente = serie_completa_real.tail(ultimos_dias_reais)

    # Trace: Histórico Real Recente
    fig_fore.add_trace(go.Scatter(
        x=serie_historico_recente.index, y=serie_historico_recente.values,
        mode="lines", name=f"Histórico Real ({ultimos_dias_reais} dias)",
        line=dict(color="#0d6efd") # Azul Bootstrap
    ))
    # Trace: Intervalo de Confiança (Área)
    fig_fore.add_trace(go.Scatter(
        x=df_forecast["Data"].append(df_forecast["Data"][::-1]), # Usa append para o índice
        y=pd.concat([df_forecast["Limite_Superior_IC95"], df_forecast["Limite_Inferior_IC95"][::-1]]),
        fill="toself", fillcolor="rgba(220, 53, 69, 0.15)", line=dict(color="rgba(255,255,255,0)"), # Vermelho Bootstrap transparente
        hoverinfo="skip", name="Intervalo de Confiança 95%"
    ))
    # Trace: Previsão ARIMA
    fig_fore.add_trace(go.Scatter(
        x=df_forecast["Data"], y=df_forecast["Fechamento_Previsto"],
        mode="lines+markers", name=f"Previsão ARIMA ({n_periods} dias)",
        line=dict(color="#dc3545", dash="dash"), marker=dict(size=4) # Vermelho Bootstrap com traço e marcadores pequenos
    ))

    fig_fore.update_layout(
        title=f"Previsão ARIMA para os Próximos {n_periods} Dias Úteis",
        xaxis_title="Data", yaxis_title="Fechamento (Pontos)", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Legenda horizontal acima
    )
    st.plotly_chart(fig_fore, use_container_width=True)

    with st.expander(f"Visualizar tabela com a previsão para os próximos {n_periods} dias úteis"):
        st.dataframe(df_forecast.style.format({"Fechamento_Previsto": "{:,.2f}", "Limite_Inferior_IC95": "{:,.2f}", "Limite_Superior_IC95": "{:,.2f}"}), use_container_width=True)

except Exception as e:
    st.error(f"Falha ao gerar previsão ARIMA: {e}")
    st.error(traceback.format_exc())


# 6. Análise de Resíduos
# ----------------------
st.markdown("---")
st.subheader("6. Análise de Resíduos do Modelo")

try:
    with st.spinner("Analisando resíduos do modelo..."):
        # Pega resíduos do último ajuste/update
        residuos = modelo_arima.resid()
        st.write("Resíduos são a diferença entre os valores reais e os valores previstos pelo modelo dentro da amostra de treino/ajuste.")

        fig_res = px.histogram(residuos, nbins=50, title="Distribuição dos Resíduos do Modelo")
        fig_res.update_layout(xaxis_title="Valor do Resíduo", yaxis_title="Frequência")
        st.plotly_chart(fig_res, use_container_width=True)

        # Testes estatísticos nos resíduos
        if len(residuos) > 3:
            stat_shapiro, p_shapiro = shapiro(residuos)
            res_mean = np.mean(residuos)
            res_std = np.std(residuos)

            col1, col2, col3 = st.columns(3)
            col1.metric(label="Média dos Resíduos", value=f"{res_mean:.2f}")
            col2.metric(label="Desvio Padrão dos Resíduos", value=f"{res_std:.2f}")
            col3.metric(label="Teste Normalidade (Shapiro) p-valor", value=f"{p_shapiro:.3f}")

            if p_shapiro > 0.05:
                st.success("O Teste de Shapiro-Wilk (p > 0.05) sugere que os resíduos **não são significativamente diferentes** de uma distribuição normal (bom sinal).")
            else:
                st.warning("O Teste de Shapiro-Wilk (p <= 0.05) sugere que os resíduos **podem não seguir** uma distribuição normal (pode indicar que o modelo não capturou toda a estrutura dos dados).")
        else:
            st.warning("Não há resíduos suficientes para realizar testes estatísticos.")

except Exception as e:
    st.warning(f"Não foi possível gerar análise de resíduos: {e}.")


# 7. Cálculo de Acurácia (MAPE no Teste Real)
# --------------------------------------------
st.markdown("---")
st.subheader("7. Avaliação da Acurácia no Conjunto de Teste")

if teste_real is not None and not teste_real.empty:
    try:
        with st.spinner(f"Gerando previsões para o período de teste ({len(teste_real)} dias)..."):
            # Previsão para o período exato do conjunto de teste
            forecast_teste, conf_int_teste = modelo_arima.predict(n_periods=len(teste_real), return_conf_int=True)

            df_comp = pd.DataFrame({
                'Real': teste_real.values,
                'Previsto': forecast_teste,
                'Limite_Inferior_IC95': conf_int_teste[:, 0],
                'Limite_Superior_IC95': conf_int_teste[:, 1]
            }, index=teste_real.index)

        st.write(f"Comparando valores reais e previstos para os últimos {len(teste_real)} dias (conjunto de teste):")

        # Função MAPE
        def mape(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            non_zero_mask = y_true != 0
            if np.sum(non_zero_mask) == 0: return np.nan
            return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

        erro_mape = mape(df_comp['Real'], df_comp['Previsto'])
        acuracia = 100 - erro_mape

        # Exibe métricas
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="MAPE (Erro Percentual Médio Absoluto)", value=f"{erro_mape:.2f}%" if not np.isnan(erro_mape) else "N/A")
        with col2:
            st.metric(label="Acurácia (100% - MAPE)", value=f"{acuracia:.2f}%" if not np.isnan(acuracia) else "N/A")

        if not np.isnan(acuracia):
            if acuracia >= 95:
                 st.success(f"Acurácia de {acuracia:.2f}% no teste real é considerada EXCELENTE.")
            elif acuracia >= 85:
                st.success(f"Acurácia de {acuracia:.2f}% no teste real é considerada BOA.")
            elif acuracia >= 70:
                st.info(f"Acurácia de {acuracia:.2f}% no teste real é ACEITÁVEL.")
            else:
                st.warning(f"Acurácia de {acuracia:.2f}% no teste real é BAIXA. O modelo pode precisar de ajustes ou reavaliação.")

        # Gráfico Comparativo no Teste
        fig_teste = go.Figure()
        fig_teste.add_trace(go.Scatter(x=df_comp.index, y=df_comp['Real'], mode='lines', name='Valor Real', line=dict(color="#0d6efd")))
        fig_teste.add_trace(go.Scatter(
            x=df_comp.index.append(df_comp.index[::-1]), # Correção aplicada
            y=pd.concat([df_comp["Limite_Superior_IC95"], df_comp["Limite_Inferior_IC95"][::-1]]),
            fill="toself", fillcolor="rgba(220, 53, 69, 0.1)", line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip", name="Intervalo Conf. 95% (Teste)"
        ))
        fig_teste.add_trace(go.Scatter(x=df_comp.index, y=df_comp['Previsto'], mode='lines', name='Valor Previsto', line=dict(color="#dc3545", dash='dot')))

        fig_teste.update_layout(title='Comparação Real vs. Previsto no Conjunto de Teste', xaxis_title='Data', yaxis_title='Fechamento (Pontos)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_teste, use_container_width=True)

        with st.expander("Visualizar tabela comparativa (teste)"):
            st.dataframe(df_comp.style.format("{:,.2f}"), use_container_width=True)

    except Exception as e:
        st.error(f"Falha ao calcular acurácia ou comparar no teste real: {e}")
        st.error(traceback.format_exc()) # Mantém traceback para erros inesperados aqui
else:
    st.warning("Avaliação de acurácia não realizada (não há dados de teste suficientes ou ocorreram erros anteriores).")

# =========================================
# --- Fim da Lógica Principal do App ---
# =========================================
st.markdown("---")
st.success("Análise e Previsão Concluídas!")
# st.balloons() # Removido, pode ser um pouco demais para uma avaliação formal
