from flask_sqlalchemy import SQLAlchemy
from flask import Flask
import config

app = Flask(__name__)
app.config.from_object(config)
db = SQLAlchemy(app)

class Modelo(db.Model):
    __tablename__ = 'modelos'  # CORRECTO
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    descripcion = db.Column(db.Text, nullable=False)
    fuente = db.Column(db.String(300), nullable=False)
    contenido_grafico = db.Column(db.String(300), nullable=True)

with app.app_context():
    db.create_all()
