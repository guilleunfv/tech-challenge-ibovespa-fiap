from flask import Flask, jsonify
import pandas as pd
import joblib
import os
import traceback
import logging
from google.cloud import bigquery, storage

# Librería pmdarima para AutoARIMA
from pmdarima import auto_arima

app = Flask(__name__)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ID = os.getenv("PROJECT_ID", "ibovespa-data-project")
BUCKET_NAME = os.getenv("BUCKET_NAME", "ibovespa-models")
BQ_DATASET = os.getenv("BQ_DATASET", "ibovespa_dataset")
BQ_TABLE = os.getenv("BQ_TABLE", "dados_historicos")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "online", "mensagem": "Acesse /treinar para iniciar o treinamento (ARIMA)."}), 200

@app.route("/treinar", methods=["GET"])
def treinar():
    try:
        logging.info("Iniciando treinamento ARIMA...")

        # 1) Leer datos desde BigQuery
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT_ID)
        query = f"""
            SELECT Data, Fechamento
            FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
            ORDER BY Data ASC
        """
        df = client.query(query).to_dataframe()

        if df.empty:
            logging.warning("Nenhum dado encontrado no BigQuery.")
            return jsonify({"status": "erro", "mensagem": "Nenhum dado encontrado no BigQuery."}), 500

        logging.info(f"Dados carregados: {df.shape[0]} filas.")
        logging.info(f"Estatísticas:\n{df.describe()}")

        # 2) Preparar datos: extraer la columna Fechamento
        # ARIMA se entrena con una serie 1D
        # Opcional: tail(1000) si quieres reducir
        serie = df["Fechamento"].astype(float)

        # 3) Entrenar modelo ARIMA automáticamente
        logging.info("Entrenando AutoARIMA, por favor espera...")
        modelo_arima = auto_arima(
            y=serie,
            start_p=1, start_q=1,
            max_p=5, max_q=5,
            seasonal=False,      # Si tuvieras estacionalidad clara, pon True
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )

        logging.info("Modelo ARIMA entrenado con éxito.")
        logging.info(f"Parámetros del modelo: {modelo_arima.get_params()}")

        # 4) Guardar modelo localmente
        local_model_path = "modelo_arima.pkl"
        joblib.dump(modelo_arima, local_model_path)
        logging.info(f"Modelo guardado en: {local_model_path}")

        # 5) Subir a GCS
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob("modelos/modelo_arima.pkl")
        blob.upload_from_filename(local_model_path)

        logging.info(f"Modelo ARIMA subido a gs://{BUCKET_NAME}/modelos/modelo_arima.pkl")

        return jsonify({
            "status": "sucesso",
            "mensagem": "Modelo ARIMA entrenado y guardado en GCS.",
            "modelo_gcs_path": f"gs://{BUCKET_NAME}/modelos/modelo_arima.pkl"
        }), 200

    except Exception as e:
        logging.error("Error en el entrenamiento de ARIMA.")
        return jsonify({
            "status": "erro",
            "mensagem": "Falha no treinamento.",
            "erro": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
