import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_cors import CORS
import asyncio
from predecir import noseCompaeEstoyCansado

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:A*Caae55122f3g3G*A-FbBAeCg4Agb15@roundhouse.proxy.rlwy.net:16253/railway'#postgresql://h0l4dmin:kwkMIlPzfWhDBjFhe2CR@hola-bd-qa.ckeqxfmdqgne.us-east-1.rds.amazonaws.com/incoporacion
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'senku'
# Obtén el puerto de la variable de entorno, o utiliza 5000 como valor por defecto
port = int(os.environ.get('PORT', 5000))

CORS(app)
# Configuración de la base de datos
db = SQLAlchemy(app)
# Configuración de la base de datos
migrate = Migrate(app, db)

# Configuración de la extensión de JWT
jwt = JWTManager(app)

# Importar rutas después de inicializar la aplicación y la base de datos
from routes.routes import *

if __name__ == '__main__':
    # Crear las tablas en la base de datos
    with app.app_context():
        db.create_all()
    asyncio.run(noseCompaeEstoyCansado())
    
    app.run(debug=True,host='0.0.0.0',port=port)
