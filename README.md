# 📈 Tech Challenge Ibovespa - FIAP

Este repositório contém o projeto desenvolvido durante o Tech Challenge da FIAP, que utiliza técnicas de ciência de dados para realizar previsões diárias do fechamento do índice IBOVESPA.

This repository contains the project developed during the FIAP Tech Challenge, using data science techniques to make daily predictions of the IBOVESPA index closing.

Este repositorio contiene el proyecto desarrollado durante el Tech Challenge de FIAP, que utiliza técnicas de ciencia de datos para realizar predicciones diarias del cierre del índice IBOVESPA.

---

## 📂 Organização do Projeto | Project Structure | Organización del Proyecto

Este repositório está estruturado em três partes principais:

This repository is structured in three main parts:

Este repositorio está estructurado en tres partes principales:

### 1. 📥 [dados-historicos-bigquery](./dados-historicos-bigquery)

Responsável pela **coleta, limpeza e armazenamento** dos dados históricos do IBOVESPA em uma base de dados no BigQuery (Google Cloud).

Responsible for **collecting, cleaning, and storing** historical IBOVESPA data in a BigQuery database (Google Cloud).

Responsable de la **recolección, limpieza y almacenamiento** de los datos históricos del IBOVESPA en una base de datos en BigQuery (Google Cloud).

### 2. 🚀 [modelo-arima-streamlit](./modelo-arima-streamlit)

Aplicação web interativa criada com **Streamlit**, apresentando previsões do fechamento do IBOVESPA utilizando o modelo **ARIMA** previamente treinado.

Interactive web application created with **Streamlit**, presenting IBOVESPA closing predictions using a previously trained **ARIMA** model.

Aplicación web interactiva creada con **Streamlit**, que presenta predicciones del cierre del IBOVESPA utilizando el modelo **ARIMA** previamente entrenado.

### 3. 🧠 [treinamento-modelo-prophet](./treinamento-modelo-prophet)

Scripts e configuração para treinamento e validação do modelo preditivo alternativo utilizando **Prophet** (Meta/Facebook).

Scripts and configuration for training and validation of an alternative predictive model using **Prophet** (Meta/Facebook).

Scripts y configuración para entrenamiento y validación del modelo predictivo alternativo utilizando **Prophet** (Meta/Facebook).

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas | Technologies and Libraries Used | Tecnologías y Bibliotecas Utilizadas

### Linguagens e Frameworks | Languages and Frameworks | Lenguajes y Frameworks
- Python
- Flask==2.3.2
- Streamlit==1.25.0
- Docker

### Bibliotecas de Ciência de Dados | Data Science Libraries | Bibliotecas de Ciencia de Datos
- numpy==1.23.5 *(versão fixa para compatibilidade com Python 3.10 | fixed version for compatibility with Python 3.10 | versión fija para compatibilidad con Python 3.10)*
- numpy<2 *(para evitar incompatibilidades com bibliotecas antigas | to avoid incompatibilities with older libraries | para evitar incompatibilidades con bibliotecas antiguas)*
- pandas==1.5.3
- matplotlib
- seaborn
- statsmodels==0.14.0
- pmdarima==2.0.3
- ARIMA
- Prophet

### Visualização e Gráficos | Visualization and Graphics | Visualización y Gráficos
- plotly==5.15.0

### Google Cloud e Serviços | Google Cloud and Services | Google Cloud y Servicios
- Google Cloud Run
- google-auth==2.22.0
- google-cloud-bigquery==3.9.0
- google-cloud-storage==2.10.0
- db-dtypes *(necessário para manipulação do BigQuery | required for BigQuery handling | necesario para manipulación de BigQuery)*

### Dependências Adicionais | Additional Dependencies | Dependencias Adicionales
- requests==2.31.0
- joblib
- gunicorn

---

## ⚙️ Como executar os projetos | How to run the projects | Cómo ejecutar los proyectos

Consulte os arquivos `README.md` específicos em cada pasta para instruções detalhadas sobre instalação e execução.

Refer to the specific `README.md` files in each folder for detailed installation and execution instructions.

Consulta los archivos `README.md` específicos en cada carpeta para obtener instrucciones detalladas sobre instalación y ejecución.

---

## 🤝 Equipe do Projeto | Project Team | Equipo del Proyecto

- Guillermo Jesus Camahuali Privat (guilleunfv@gmail.com)
- Rosicleia Cavalcante Mota (rosim.controladoria@gmail.com)
- Kelly Priscilla Matos Campos (Kellyp.mcampos@hotmail.com)

---

## 📜 Licença | License | Licencia

Este projeto está licenciado sob a licença MIT - consulte o arquivo [LICENSE](./LICENSE) para mais detalhes.

This project is licensed under the MIT license - see the [LICENSE](./LICENSE) file for more details.

Este proyecto está licenciado bajo la licencia MIT - consulta el archivo [LICENSE](./LICENSE) para más detalles.


