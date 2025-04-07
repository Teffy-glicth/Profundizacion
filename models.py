from flask_sqlalchemy import SQLAlchemy
from flask import Flask
import config


db = SQLAlchemy()

class Modelo(db.Model):
    __tablename__ = 'modelos'  
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    descripcion = db.Column(db.Text, nullable=False)
    fuente = db.Column(db.String(300), nullable=False)
    contenido_grafico = db.Column(db.String(300), nullable=True)


