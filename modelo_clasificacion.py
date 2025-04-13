import pandas as pd 
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

df_resultados = None

def predecir_desde_excel(archivo):
    # Leer datos
    df = pd.read_excel(archivo)
    
    # Normalizar nombres de columnas
    df.columns = df.columns.str.lower()  
    df.columns = df.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')  
    
    # Preparar datos
    X = df[['tiempo_suscripcion', 'num_quejas', 'uso_datos']]
    y = df['abandono']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    modelo = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.15,
        max_depth=4,
        random_state=42
    )
    modelo.fit(X_train, y_train)
    
    # Realizar predicciones
    predicciones = modelo.predict(X)
    
    # Crear un nuevo DataFrame solo con las columnas que queremos mostrar
    df_mostrar = df[['tiempo_suscripcion', 'num_quejas', 'uso_datos']]
    df_mostrar['Prediccion'] = predicciones
    
    global df_resultados
    df_resultados = df_mostrar  # Guardamos solo las columnas seleccionadas
    
    return df_mostrar

def exportar_resultados():
    global df_resultados
    if df_resultados is not None:
        ruta = "resultados.csv"
        df_resultados.to_csv(ruta, index=False)
        return ruta
    return None