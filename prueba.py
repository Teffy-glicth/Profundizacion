from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://modelos_db_user:lZUg1NORsgAtBIrBJR70jWwcOUldbsow@dpg-cvq1ngjuibrs73879030-a/modelos_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

@app.route('/')
def test_db():
    try:
        
        db.session.execute('SELECT 1')
        return 'Conexión exitosa con la base de datos en Render'
    except Exception as e:
        return f'Error al conectar con la base de datos: {e}'
