FROM python:3.10

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivos necessários
COPY train_prophet.py ./train_prophet.py
COPY requirements.txt ./requirements.txt

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --upgrade pip \
    && pip install --upgrade cmdstanpy \
    && python -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.31.0')"

# Definir porta
EXPOSE 8080

# Comando para rodar a aplicação
CMD ["gunicorn", "-b", "0.0.0.0:8080", "train_prophet:app"]
