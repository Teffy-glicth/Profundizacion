from app import app, db
from models import Modelo


with app.app_context():
    nuevo_modelo = Modelo(
        nombre="Naive Bayes",
        descripcion="""El Naive Bayes es un algoritmo de aprendizaje supervisado basado en el Teorema de Bayes, utilizado principalmente para clasificación. Funciona calculando la probabilidad de que un dato pertenezca a una clase específica, utilizando la suposición de independencia condicional entre características (de ahí el término "ingenuo"), lo que simplifica los cálculos al considerar que cada atributo contribuye de manera independiente a la probabilidad final Características: 
        Simplicidad: Fácil de implementar y entender, ideal para proyectos rápidos o como punto de referencia comparativo. Eficiencia con grandes volúmenes de datos: Procesa rápidamente conjuntos masivos, como textos o registros médicos, sin requerir alta capacidad computacional. 
        Rendimiento en datos categóricos: Destaca en aplicaciones con variables discretas (ejemplo: presencia de palabras en correos electrónicos). Robustez ante características irrelevantes: Mitiga el impacto de variables poco informativas en las predicciones finales. 
        Tipos de modelos: Incluye variantes como Gaussiano (para datos continuos), Multinomial (conteos de frecuencias) y Bernoulli (datos binarios). Limitaciones: La suposición de independencia entre características puede reducir precisión en problemas complejos, y enfrenta problemas con categorías no vistas en entrenamiento (requiere técnicas como alisamiento). 
        Aplicaciones comunes: Clasificación de documentos, sistemas de recomendación, detección de spam y análisis predictivo en medicina""",
         fuente="""https://aprendeia.com/algoritmo-naive-bayes-machine-learning/ https://forum.huawei.com/enterprise/intl/es/m%C3%A9todo-supervisado-clasificaci%C3%B3n-naive-bayes/thread/872753-100757?isInitURL=true""",
        contenido_grafico="img/naivebayes.png"
    )
    db.session.add(nuevo_modelo)
    db.session.commit()
    print("Modelo insertado correctamente.")