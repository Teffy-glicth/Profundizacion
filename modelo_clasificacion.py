import pandas as pd 
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.impute import SimpleImputer

df_resultados = None

def predecir_desde_excel(archivo):
    # Leer datos
    df = pd.read_excel(archivo)
    
    # Normalizar nombres de columnas
    df.columns = df.columns.str.lower()  
    df.columns = df.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')  
    
    # Preparar datos y target
    X = df[['tiempo_suscripcion', 'num_quejas', 'uso_datos']]
    y = df['abandono']  # Movido aquí arriba
    
    # Características más específicas para el problema
    # Ajustar estos umbrales según los patrones de tus datos
    X['quejas_graves'] = (X['num_quejas'] >= 7).astype(int)  # Bajado de 10 a 7
    X['tiempo_corto'] = (X['tiempo_suscripcion'] <= 40).astype(int)  # Subido de 30 a 40
    X['uso_bajo'] = (X['uso_datos'] <= 13.5).astype(int)  # Ajustado a 13.5
    X['riesgo_abandono'] = ((X['num_quejas'] >= 8) & (X['tiempo_suscripcion'] <= 50)).astype(int)
    X['cliente_estable'] = ((X['tiempo_suscripcion'] >= 70) & (X['num_quejas'] <= 5) & (X['uso_datos'] >= 15)).astype(int)
    
    # Características compuestas
    X['indice_satisfaccion'] = (X['tiempo_suscripcion'] * X['uso_datos']) / (X['num_quejas'] + 1)
    X['tasa_quejas'] = X['num_quejas'] / (X['tiempo_suscripcion'] + 1)
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Modelo ajustado
    modelo = GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.005,
        max_depth=3,
        min_samples_split=15,
        min_samples_leaf=5,
        subsample=0.7,
        max_features='sqrt',
        random_state=42
    )
    
    # Entrenamiento con más iteraciones
    modelo.fit(X_train, y_train)
    
    # Evaluación
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Generar gráfico de matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    
    # Guardar gráfico en memoria
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Realizar predicciones en todo el conjunto
    predicciones = modelo.predict(X)
    df_mostrar = df[['tiempo_suscripcion', 'num_quejas', 'uso_datos']]
    df_mostrar['Prediccion'] = predicciones
    
    global df_resultados
    df_resultados = df_mostrar
    
    return df_mostrar, {
        'accuracy': accuracy,
        'report': report,
        'plot_url': plot_url
    }

def exportar_resultados():
    global df_resultados
    if df_resultados is not None:
        ruta = "resultados.csv"
        df_resultados.to_csv(ruta, index=False)
        return ruta
    return None