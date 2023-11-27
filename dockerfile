# Usa una imagen de Python como base
FROM python:3.11

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el archivo de requerimientos al contenedor
COPY requirements.txt /app/

# Instala dependencias del sistema operativo (si es necesario)
# RUN apt-get update && apt-get install -y libpq-dev

# Imprime el contenido del archivo requirements.txt (para verificar)
RUN cat requirements.txt

# Instala los requerimientos
RUN pip install --no-cache-dir -r requirements.txt

# # Imprime información sobre las dependencias instaladas
# RUN pip show -r requirements.txt

# Copia el contenido del directorio actual al contenedor en /app
COPY . /app

# Expone el puerto 5000 para que Flask pueda ser accedido desde fuera del contenedor
EXPOSE 5000

# Define el comando por defecto que se ejecutará cuando se inicie el contenedor
CMD ["python", "main.py"]