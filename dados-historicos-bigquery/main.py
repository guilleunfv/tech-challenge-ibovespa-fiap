from flask import Flask, jsonify
import os
import traceback

import requests
import pandas as pd
from google.cloud import bigquery

import io
from datetime import datetime

app = Flask(__name__)

# ---------------------------------------------------------------------------------------
# 1. Rota de Teste de Conexão à Internet
#    Verifica se o Cloud Run consegue chegar na internet em geral.
# ---------------------------------------------------------------------------------------
@app.route("/test-connection", methods=["GET"])
def test_connection():
    test_url = "https://www.google.com"
    try:
        r = requests.get(test_url, timeout=5)
        return jsonify({
            "msg": "Teste de conexão bem-sucedido",
            "url": test_url,
            "status_code": r.status_code
        }), 200
    except Exception as e:
        app.logger.error("Erro em test_connection: %s", str(e))
        traceback.print_exc()
        return jsonify({
            "msg": f"Não foi possível conectar em {test_url}",
            "error": str(e)
        }), 500

# ---------------------------------------------------------------------------------------
# 2. Função para obter dados do IBOVESPA
# ---------------------------------------------------------------------------------------
def obter_dados_ibovespa(url):
    """
    Faz o download e retorna o DataFrame de dados históricos do IBOVESPA
    a partir da URL fornecida.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/123.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        app.logger.error("Erro ao obter dados de %s: %s", url, str(e))
        traceback.print_exc()
        return None

# ---------------------------------------------------------------------------------------
# 3. Função para limpar dados
# ---------------------------------------------------------------------------------------
def limpar_dados_ibovespa(df):
    """
    Faz transformações e limpeza de dados no DataFrame.
    Renomeia colunas para português, converte tipos e remove linhas inválidas.
    """
    if df is None or df.empty:
        app.logger.error("DataFrame vazio ou inválido.")
        return None

    try:
        # Renomear colunas para português
        df.rename(columns={
            'Date': 'Data',
            'Close': 'Fechamento',
            'High': 'Maximo',
            'Low': 'Minimo',
            'Open': 'Abertura',
            'Volume': 'Volume'
        }, inplace=True)

        # Converter a coluna 'Data' para datetime
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        # Eliminar linhas com datas inválidas
        df.dropna(subset=['Data'], inplace=True)

        # Converter colunas numéricas e preencher valores inválidos com 0
        for col in ['Fechamento', 'Abertura', 'Maximo', 'Minimo', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df
    except Exception as e:
        app.logger.error("Erro durante a limpeza de dados: %s", str(e))
        traceback.print_exc()
        return None

# ---------------------------------------------------------------------------------------
# 4. Função para carregar dados no BigQuery
# ---------------------------------------------------------------------------------------
def carregar_dados_bigquery(df, project_id, dataset_id, table_id):
    """
    Carrega um DataFrame para uma tabela no BigQuery, criando a tabela se necessário.
    """
    if df is None or df.empty:
        app.logger.error("O DataFrame está vazio. Não há dados para carregar no BigQuery.")
        return

    try:
        client = bigquery.Client(project=project_id)
        table_ref = client.dataset(dataset_id).table(table_id)

        # Verifica se a tabela existe; se não existir, cria com o schema.
        try:
            client.get_table(table_ref)
        except Exception:
            schema = [
                bigquery.SchemaField("Data", "DATE"),
                bigquery.SchemaField("Fechamento", "FLOAT"),
                bigquery.SchemaField("Abertura", "FLOAT"),
                bigquery.SchemaField("Maximo", "FLOAT"),
                bigquery.SchemaField("Minimo", "FLOAT"),
                bigquery.SchemaField("Volume", "FLOAT"),
            ]
            table = bigquery.Table(table_ref, schema=schema)
            table = client.create_table(table)
            app.logger.info("Tabela criada: %s", table.table_id)

        # Configuração do job de carregamento
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            source_format=bigquery.SourceFormat.CSV,
        )
        # Carrega o DataFrame para a tabela
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  # Espera o job terminar

        app.logger.info(
            "Carregadas %d linhas em %s.%s.%s",
            job.output_rows, project_id, dataset_id, table_id
        )
    except Exception as e:
        app.logger.error("Erro ao carregar dados no BigQuery: %s", str(e))
        traceback.print_exc()

# ---------------------------------------------------------------------------------------
# 5. Rota principal
# ---------------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def main():
    """
    Ponto de entrada da aplicação no Cloud Run.
    Faz o download do CSV do IBOVESPA, limpa e carrega os dados no BigQuery.
    """
    try:
        url_ibovespa = "https://storage.googleapis.com/ibovespa-data-tech-challenge/ibovespa/ibovespa_data.csv"
        df_ibovespa = obter_dados_ibovespa(url_ibovespa)

        if df_ibovespa is None:
            return "Erro: Não foi possível obter os dados do IBOVESPA.", 500

        df_limpio = limpar_dados_ibovespa(df_ibovespa)
        if df_limpio is None:
            return "Erro: Não foi possível limpar os dados.", 500

        project_id = os.environ.get("PROJECT_ID", "ibovespa-data-project")
        dataset_id = "ibovespa_dataset"
        table_id = "dados_historicos"

        carregar_dados_bigquery(df_limpio, project_id, dataset_id, table_id)

        return "Dados carregados com sucesso no BigQuery.", 200

    except Exception as e:
        app.logger.error("Erro geral em main(): %s", str(e))
        traceback.print_exc()
        return "Ocorreu um erro inesperado.", 500

# ---------------------------------------------------------------------------------------
# 6. Execução da App em modo local
#    (No Cloud Run, a variável de ambiente PORT é injetada automaticamente)
# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
