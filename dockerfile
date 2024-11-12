# Use uma imagem base do Python com TensorFlow e outras dependências
FROM tensorflow/tensorflow:2.11.0

# Instale dependências adicionais
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Defina o diretório de trabalho
WORKDIR /app

# Copie todos os arquivos para o contêiner
COPY . .

# Instale as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponha a porta que o serviço usa
EXPOSE 5000

# Comando para iniciar a aplicação
CMD ["python", "main.py"]
