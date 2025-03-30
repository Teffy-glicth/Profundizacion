from flask import Flask, render_template, request
from datetime import datetime
import re
from linearRegression import calculateGrade
from linearRegression import grafica

app = Flask(__name__)

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