# 1. Imagem base oficial do Python (versão 3.12 leve)
FROM python:3.12-slim

# 2. Define a pasta de trabalho dentro do container
WORKDIR /app

# 3. Copia apenas a lista de dependências primeiro e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copia o código da API (com a página web) e o modelo treinado
COPY app/ ./app/
COPY model/ ./model/

# 5. Documenta que o container escuta na porta 8000
EXPOSE 8000

# 6. Comando que inicia a API quando o container ligar
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]