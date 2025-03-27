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
import traceback # Para logs mais detalhados
import copy # Para copiar o modelo
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
# Verifica se as credenciais existem primeiro
if "google_credentials" not in st.secrets:
    st.error("Erro Crítico: Credenciais do Google ('google_credentials') não configuradas nos Secrets do Streamlit.")
    st.stop()

# Lê os parâmetros do projeto e nomes de recursos
PROJECT_ID = st.secrets.get("PROJECT_ID")
BUCKET_NAME = st.secrets.get("BUCKET_NAME")
BLOB_NAME = "modelos/modelo_arima.pkl"  # Caminho fixo do modelo no GCS
BQ_DATASET = st.secrets.get("BQ_DATASET")
BQ_TABLE = st.secrets.get("BQ_TABLE")

# Valida se todos os parâmetros foram lidos (exceto credenciais que já verificamos)
if not all([PROJECT_ID, BUCKET_NAME, BQ_DATASET, BQ_TABLE]):
    missing_params = [k for k, v in {"PROJECT_ID": PROJECT_ID, "BUCKET_NAME": BUCKET_NAME, "BQ_DATASET": BQ_DATASET, "BQ_TABLE": BQ_TABLE}.items() if not v]
    st.error(f"Erro Crítico: Parâmetros não configurados nos Secrets do Streamlit: {', '.join(missing_params)}")
    st.stop()

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
    """Carrega as credenciais a partir dos secrets."""
    try:
        creds_dict = dict(st.secrets["google_credentials"])
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
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
        return None # Erro já reportado em carregar_credenciais_google

    try:
        client = storage.Client(project=project_id, credentials=credentials)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
             st.error(f"Erro: Modelo não encontrado no GCS em gs://{bucket_name}/{blob_name}")
             return None

        st.info(f"Baixando modelo de gs://{bucket_name}/{blob_name}...")
        model_bytes = blob.download_as_bytes()
        modelo_arima = joblib.load(BytesIO(model_bytes))
        st.success("Modelo ARIMA carregado com sucesso!")
        return modelo_arima
    except Exception as e:
        st.error(f"Erro ao baixar/carregar modelo ARIMA do GCS: {e}")
        st.error(traceback.format_exc())
        return None

# =========================================
# Função para Carregar Dados Reais do BigQuery (Cache de Dados)
# =========================================
@st.cache_data(ttl=3600, show_spinner="Carregando dados reais do BigQuery...")
def carregar_dados_reais_bq(project_id, bq_dataset, bq_table):
    """Carrega dados históricos reais (Data, Fechamento) do BigQuery."""
    st.info("Iniciando carregamento de dados BQ...") # Log 1
    credentials_local = carregar_credenciais_google()
    if credentials_local is None:
         st.error("Falha ao carregar credenciais locais.") # Log erro
         return None

    # --- LOG DETALHADO DAS CREDENCIAIS ---
    try:
        st.info(f"Tentando usar credenciais para Service Account: {credentials_local.service_account_email}") # Log 2
        st.info(f"Credenciais válidas: {credentials_local.valid}") # Log 3
        # st.info(f"Escopos da credencial: {credentials_local.scopes}") # Log 4 (pode ser muito verbose)
    except Exception as cred_e:
        st.warning(f"Não foi possível obter detalhes da credencial: {cred_e}")
    # --- FIM DO LOG DETALHADO ---

    try:
        # --- Tente criar o cliente SEM o project_id explícito ---
        # Às vezes, o cliente infere melhor o projeto das credenciais.
        # client = bigquery.Client(project=project_id, credentials=credentials_local)
        st.info("Criando cliente BigQuery SEM project_id explícito...") # Log 5
        client = bigquery.Client(credentials=credentials_local)
        st.info(f"Cliente BigQuery criado. Projeto inferido: {client.project}") # Log 6 - Verifica o projeto usado
        # ---------------------------------------------------------

        # Query original (assumindo colunas Data e Fechamento)
        nome_coluna_data = "Data"
        nome_coluna_fechamento = "Fechamento"
        table_id_completo = f"`{client.project}.{bq_dataset}.{bq_table}`" # Usa projeto inferido

        query = f"""
            SELECT {nome_coluna_data}, {nome_coluna_fechamento}
            FROM {table_id_completo}
            WHERE {nome_coluna_fechamento} IS NOT NULL AND DATE({nome_coluna_data}) <= CURRENT_DATE()
            ORDER BY {nome_coluna_data} ASC
        """

        st.info(f"Executando query na tabela '{bq_table}': {query}") # Log 7 (confirma query)

        df = client.query(query).to_dataframe()

        st.success(f"Query executada com sucesso para {table_id_completo}.")

        df.rename(columns={nome_coluna_data: 'Data', nome_coluna_fechamento: 'Fechamento'}, inplace=True)

        if df.empty:
            st.warning(f"A query para {table_id_completo} não retornou dados.")
            return pd.DataFrame()

        df = df.drop_duplicates(subset='Data', keep='last') # Mover para depois do rename se necessário
        return df

    except Exception as e:
        st.error(f"Erro DENTRO da execução da query ou processamento: {e}") # Log erro
        st.error(traceback.format_exc())
        return pd.DataFrame()

# =========================================
# --- Início da Lógica Principal do App ---
# =========================================

# 1. Carregar Modelo
# ------------------
modelo_arima_original = baixar_modelo_arima_gcs(BUCKET_NAME, BLOB_NAME, PROJECT_ID)

if modelo_arima_original is None:
    st.error("Não foi possível carregar o modelo ARIMA. Verifique os logs e as configurações.")
    st.stop()

# Tenta copiar o modelo para evitar modificar o original cacheado durante updates
try:
    modelo_arima = copy.deepcopy(modelo_arima_original)
    st.info(f"Modelo ARIMA carregado ({type(modelo_arima).__name__}). Parâmetros originais: {modelo_arima.get_params()}")
except Exception as e:
    st.warning(f"Não foi possível fazer deepcopy do modelo ({e}). Usando referência original (cuidado com 'update').")
    modelo_arima = modelo_arima_original


# 2. Carregar Dados Reais
# -----------------------
df_real_raw = carregar_dados_reais_bq(PROJECT_ID, BQ_DATASET, BQ_TABLE)

if df_real_raw is None or df_real_raw.empty:
    st.error("Não foi possível carregar os dados reais do BigQuery. Verifique os logs e as configurações.")
    st.stop()


# 3. Pré-processar Dados (Frequência, Preenchimento)
# -------------------------------------------------
st.markdown("---")
st.subheader("3. Processamento e Visualização dos Dados Reais")
st.write("Ajustando frequência para dias úteis e preenchendo valores ausentes...")

df_real = df_real_raw.copy()
try:
    df_real = df_real.set_index('Data')
    # Verifica se o índice é monotônico crescente antes de asfreq
    if not df_real.index.is_monotonic_increasing:
        st.warning("Índice de datas não é monotônico crescente após remoção de duplicatas. Reordenando...")
        df_real = df_real.sort_index()

    df_real = df_real.asfreq('B') # 'B' para dias úteis (seg-sex)
    n_preenchidos = df_real['Fechamento'].isnull().sum()
    if n_preenchidos > 0:
        st.info(f"Preenchendo {n_preenchidos} valores ausentes (fins de semana/feriados) com o método 'ffill'...")
        df_real = df_real.fillna(method='ffill')
        # Verifica se ainda há NaNs no início
        if df_real['Fechamento'].isnull().any():
             st.warning("Ainda existem valores NaN após 'ffill' (provavelmente no início da série). Preenchendo com 'bfill'...")
             df_real = df_real.fillna(method='bfill') # Preenche o início se necessário
    df_real = df_real.reset_index() # Volta Data a ser coluna
    st.success("Frequência ajustada e dados preenchidos.")
except ValueError as e:
    st.error(f"Erro ao ajustar a frequência dos dados: {e}. Verifique a consistência das datas no BigQuery. Continuando com dados originais ordenados.")
    df_real = df_real_raw.sort_values("Data").copy() # Usa dados brutos ordenados como fallback

# Garantir tipo datetime final
df_real["Data"] = pd.to_datetime(df_real["Data"])

st.write(f"**Período dos dados:** {df_real['Data'].min().strftime('%Y-%m-%d')} a {df_real['Data'].max().strftime('%Y-%m-%d')}")
st.write("**Últimos 5 registros processados:**")
st.dataframe(df_real.tail(), use_container_width=True)

# Gráfico Histórico Real
fig_hist = px.line(
    df_real, x="Data", y="Fechamento",
    title="Série Histórica Real do IBOVESPA (BigQuery Processado)",
    labels={"Data": "Data", "Fechamento": "Fechamento (pts)"},
)
fig_hist.update_layout(height=450)
st.plotly_chart(fig_hist, use_container_width=True)
st.markdown("*(Gráfico interativo: use o mouse para dar zoom ou selecionar períodos)*")


# 4. Dividir Dados para Treino/Teste e Atualizar Modelo
# ----------------------------------------------------
st.markdown("---")
st.subheader("4. Preparação para Previsão e Teste")

n_test = 30 # Dias reservados para teste (não usados na atualização)
if len(df_real) > n_test:
    serie_completa_real = df_real.set_index('Data')['Fechamento']
    # Dados de treino: todos exceto os últimos n_test dias
    treino_real = serie_completa_real.iloc[:-n_test]
    # Dados de teste: os últimos n_test dias
    teste_real = serie_completa_real.iloc[-n_test:]
    st.write(f"Dados divididos: **{len(treino_real)}** dias para ajuste/atualização do modelo, **{len(teste_real)}** dias para teste de acurácia.")

    # Opção para atualizar o modelo com os dados de treino recentes
    atualizar = st.checkbox("Atualizar modelo com dados de treino recentes antes de prever?", value=True,
                            help="Ajusta o modelo carregado aos dados mais recentes do conjunto de treino. Recomendado para previsões mais realistas, mas altera ligeiramente o modelo original.")
    if atualizar:
        try:
            st.info("Atualizando modelo com dados de treino...")
            # Usamos .update() que é mais eficiente que .fit() se o modelo já está treinado
            modelo_arima.update(treino_real.values, maxiter=50) # Limita iterações por segurança
            st.success("Modelo atualizado com sucesso.")
            st.info(f"Novos parâmetros (podem ser iguais aos originais se já estava ótimo): {modelo_arima.get_params()}")
        except Exception as e:
            st.warning(f"Falha ao atualizar o modelo: {e}. A previsão usará o modelo como carregado/copiado.")
            st.warning(traceback.format_exc())
    else:
        st.info("Modelo não será atualizado. A previsão usará o modelo conforme carregado/copiado.")

else:
    st.warning("Não há dados reais suficientes para criar um conjunto de teste separado. Apenas a previsão será gerada.")
    treino_real, teste_real = None, None
    serie_completa_real = df_real.set_index('Data')['Fechamento'] # Usa tudo para prever a partir do fim


# 5. Gerar e Visualizar Previsões
# -------------------------------
st.markdown("---")
st.subheader("5. Previsão Futura com ARIMA")

n_periods = st.slider("Selecione o número de dias úteis para prever:", min_value=10, max_value=180, value=60, step=10)

try:
    # Previsão a partir do fim da série completa (treino+teste ou tudo se não houver teste)
    st.info(f"Gerando previsão para os próximos {n_periods} dias úteis...")
    forecast, conf_int = modelo_arima.predict(n_periods=n_periods, return_conf_int=True)

    ultima_data_real = serie_completa_real.index[-1]
    datas_futuras = pd.date_range(start=ultima_data_real + pd.Timedelta(days=1),
                                  periods=n_periods, freq='B') # Frequência Dias Úteis 'B'

    df_forecast = pd.DataFrame({
        "Data": datas_futuras,
        "Fechamento_Previsto": forecast,
        "Lower_CI": conf_int[:, 0],
        "Upper_CI": conf_int[:, 1]
    })

    # Gráfico de Previsão
    fig_fore = go.Figure()
    ultimos_dias_reais = st.slider("Dias de histórico real para exibir no gráfico de previsão:", 50, 500, 200, step=50)
    # Pega os últimos N pontos da série completa real
    serie_historico_recente = serie_completa_real.tail(ultimos_dias_reais)

    fig_fore.add_trace(go.Scatter(
        x=serie_historico_recente.index,
        y=serie_historico_recente.values,
        mode="lines", name=f"Histórico Real ({ultimos_dias_reais} dias)",
        line=dict(color="#007bff") # Azul mais vibrante
    ))
    fig_fore.add_trace(go.Scatter(
        x=pd.concat([df_forecast["Data"], df_forecast["Data"][::-1]]),
        y=pd.concat([df_forecast["Upper_CI"], df_forecast["Lower_CI"][::-1]]),
        fill="toself", fillcolor="rgba(255, 0, 0, 0.15)", line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip", name="Intervalo de Confiança 95%"
    ))
    fig_fore.add_trace(go.Scatter(
        x=df_forecast["Data"], y=df_forecast["Fechamento_Previsto"],
        mode="lines", name=f"Previsão ARIMA ({n_periods} dias)",
        line=dict(color="#dc3545", dash="dash") # Vermelho com traço
    ))

    fig_fore.update_layout(
        title=f"Previsão ARIMA para os Próximos {n_periods} Dias Úteis",
        xaxis_title="Data", yaxis_title="Fechamento (pts)", height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01) # Posiciona legenda
    )
    st.plotly_chart(fig_fore, use_container_width=True)

    st.write(f"**Tabela com a previsão para os próximos {n_periods} dias úteis:**")
    st.dataframe(df_forecast.style.format({"Fechamento_Previsto": "{:.2f}", "Lower_CI": "{:.2f}", "Upper_CI": "{:.2f}"}), use_container_width=True)

except Exception as e:
    st.error(f"Falha ao gerar previsão ARIMA: {e}")
    st.error(traceback.format_exc())


# 6. Análise de Resíduos
# ----------------------
st.markdown("---")
st.subheader("6. Análise de Resíduos do Modelo")

try:
    # Pega resíduos do último fit/update (se foi feito)
    residuos = modelo_arima.resid()
    st.write("Analisando resíduos do último ajuste/atualização do modelo:")

    fig_res = px.histogram(residuos, nbins=40, title="Distribuição dos Resíduos")
    fig_res.update_layout(xaxis_title="Valor do Resíduo", yaxis_title="Frequência")
    st.plotly_chart(fig_res, use_container_width=True)

    if len(residuos) > 3:
        stat, p_value = shapiro(residuos)
        res_mean = np.mean(residuos)
        res_std = np.std(residuos)
        st.metric(label="Média dos Resíduos", value=f"{res_mean:.2f}")
        st.metric(label="Desvio Padrão dos Resíduos", value=f"{res_std:.2f}")
        st.metric(label="Teste de Normalidade (Shapiro) p-valor", value=f"{p_value:.3f}")
        if p_value > 0.05:
            st.success("Sugestão: Os resíduos parecem ser normalmente distribuídos (p > 0.05).")
        else:
            st.warning("Sugestão: Os resíduos podem NÃO ser normalmente distribuídos (p <= 0.05).")
    else:
        st.warning("Não há resíduos suficientes para realizar o teste de normalidade.")

except Exception as e:
    st.warning(f"Não foi possível gerar análise de resíduos: {e}. (O modelo pode não ter sido ajustado/atualizado ou não suporta .resid() neste estado).")


# 7. Cálculo de Acurácia (MAPE no Teste Real)
# --------------------------------------------
st.markdown("---")
st.subheader("7. Acurácia do Modelo no Conjunto de Teste Real")

if teste_real is not None and not teste_real.empty:
    try:
        st.info(f"Gerando previsões para o período de teste ({len(teste_real)} dias)...")
        # Previsão para o período exato do conjunto de teste
        forecast_teste, conf_int_teste = modelo_arima.predict(n_periods=len(teste_real), return_conf_int=True)

        df_comp = pd.DataFrame({
            'Real': teste_real.values,
            'Previsto': forecast_teste,
            'Lower_CI': conf_int_teste[:, 0],
            'Upper_CI': conf_int_teste[:, 1]
        }, index=teste_real.index)

        # Função MAPE
        def mape(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            non_zero_mask = y_true != 0
            if np.sum(non_zero_mask) == 0: return np.nan # Evita erro se todos y_true forem 0
            return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

        erro_mape = mape(df_comp['Real'], df_comp['Previsto'])
        acuracia = 100 - erro_mape

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="MAPE (Erro Percentual Médio Absoluto)", value=f"{erro_mape:.2f}%" if not np.isnan(erro_mape) else "N/A")
        with col2:
            st.metric(label="Acurácia (100% - MAPE)", value=f"{acuracia:.2f}%" if not np.isnan(acuracia) else "N/A")

        if not np.isnan(acuracia):
            if acuracia > 85:
                st.success("Acurácia no teste real considerada ALTA.")
            elif acuracia > 70:
                st.info("Acurácia no teste real considerada ACEITÁVEL.")
            else:
                st.warning("Acurácia no teste real considerada BAIXA. O modelo pode precisar de reavaliação.")

        # Gráfico Comparativo no Teste
        st.write("**Comparação Detalhada no Conjunto de Teste:**")
        fig_teste = go.Figure()
        fig_teste.add_trace(go.Scatter(x=df_comp.index, y=df_comp['Real'], mode='lines', name='Valor Real', line=dict(color="#007bff")))
        fig_teste.add_trace(go.Scatter(
            x=df_comp.index.append(df_comp.index[::-1]),
            y=pd.concat([df_comp["Upper_CI"], df_comp["Lower_CI"][::-1]]),
            fill="toself", fillcolor="rgba(255, 0, 0, 0.1)", line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip", name="Intervalo Conf. Teste"
        ))
        fig_teste.add_trace(go.Scatter(x=df_comp.index, y=df_comp['Previsto'], mode='lines', name='Valor Previsto', line=dict(color="#dc3545", dash='dot')))

        fig_teste.update_layout(title='Comparação Real vs. Previsto no Conjunto de Teste', xaxis_title='Data', yaxis_title='Fechamento (pts)', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig_teste, use_container_width=True)

        st.write("**Tabela Comparativa (Teste):**")
        st.dataframe(df_comp.style.format("{:.2f}"), use_container_width=True)

    except Exception as e:
        st.error(f"Falha ao calcular acurácia ou comparar no teste real: {e}")
        st.error(traceback.format_exc())
else:
    st.warning("Não foi possível calcular a acurácia pois não há dados de teste suficientes ou ocorreram erros anteriores.")

# =========================================
# --- Fim da Lógica Principal do App ---
# =========================================
st.markdown("---")
st.success("Análise e Previsão Concluídas!")
st.balloons()
