from flask import Flask, render_template, request
from datetime import datetime
import re
import numpy as np
from linearRegression import calculateGrade
from linearRegression import grafica
from LogisticRegression import encoder_Party, encoder_Region, scaler, model, generate_plot
from models import Modelo, db
import config
from flask import render_template, request, jsonify, send_file
from modelo_clasificacion import predecir_desde_excel, exportar_resultados


app = Flask(__name__)
app.config.from_object(config)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()

    match_object = re.fullmatch("[a-zA-Z]+", name)

    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "friend"

    content = f"Hello there, {clean_name} ! Hour: {now}"
    return content

@app.route("/example/")
def exampleHTML():
    return render_template("example.html")  

@app.route("/linearRegression/", methods =["GET", "POST"])
def linearRegression () :
    calculateResult = None
    plot_url = None
    if request.method == "POST":
        size = float (request.form["size"])
        calculateResult = predict_price(size)
        plot_url= graphic(size, calculateResult)

    return render_template("linearRegressionGrades.html", result = calculateResult, plot_url= plot_url)

@app.route("/mindmeister/")
def mindmeisterHTML():
    return render_template("mindmeister.html")

@app.route("/LogisticRegression/", methods =["GET", "POST"])
def LogisticRegression () :
    result = None
    plot_url = None

    if request.method == "POST":
        try:
            age = int(request.form ["Age"])
            region = request.form ["Region"]
            previous_answer = request.form ["Previous_Answer"]
            income = float (request.form ["Income"])

            region_encoded =encoder_Region.transform ([[region]]).toarray()
            party_encoded = encoder_Party.transform([[previous_answer]]).toarray()
            income_scaled = scaler.transform([[income]])

            input_data = np.hstack ([[age], income_scaled.flatten(), region_encoded.flatten(), party_encoded.flatten()]).reshape (1, -1)
            result = model.predict (input_data)[0]
            plot_url = generate_plot()

        except Exception as e:
            print ("Error: ", e)

    return render_template("LogisticRegression.html", result = result, plot_url= plot_url)

db.init_app(app)
with app.app_context():
    db.create_all()

@app.route("/modelos/")
def lista_modelos():
    modelos = Modelo.query.all()
    return render_template("modelo.html", modelos=modelos)

@app.route("/modelos/<int:modelo_id>")
def detalle_modelo(modelo_id):
    modelo = Modelo.query.get_or_404(modelo_id)
    return render_template("detalles_modelo.html", modelo=modelo)
@app.route('/clasificacion')
def clasificacion():
    return render_template('clasificacion.html')
@app.route("/cargar")
def cargar_modelo():
    return render_template("cargar_modelo.html")

@app.route("/predecir", methods=["POST"])
def predecir():
    archivo = request.files["archivo"]
    try:
        df, metricas = predecir_desde_excel(archivo)
        return render_template("resultados.html", 
                            tabla=df.to_html(classes="table table-bordered"),
                            accuracy=metricas['accuracy'],
                            report=metricas['report'],
                            plot_url=metricas['plot_url'],
                            mensaje="Modelo entrenado y predicciones realizadas con éxito")
    except Exception as e:
        return render_template("resultados.html", 
                            error=f"Error al procesar el archivo: {str(e)}")


@app.route("/exportar")
def exportar():
    ruta = exportar_resultados()
    if ruta:
        return send_file(ruta, as_attachment=True)
    return redirect("/")

