import numpy as np # arreglos numericos
import pandas as pd # datos en forma de dataframe
import matplotlib.pyplot as plt # grafica
import base64
import io
from sklearn.model_selection import train_test_split #dividir datos en prueba y entrenamiento
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


np.random.seed (42) #fijar los numeros random 


data = {
    "Age": np.random.randint (18, 79, 80),
    "Region": np.random.choice (["North", "South", "East", "West"], 80),
    "Previous_Answer": np.random.choice (["Partido liberal", "Pacto histórico", "Centro democrático", "Partido Verde"], 80),
    "Income": np.random.randint (1000000, 10000000, 80),
    "Vote": np.random.choice ([0,1],80)
}

df = pd.DataFrame (data)

encoder_Region = OneHotEncoder (drop = "first") #Elimina la primera linea por multicolinealidad
encoder_Party = OneHotEncoder (drop = "first")

region_encoded = encoder_Region.fit_transform (df[["Region"]]).toarray() # transforma las regiones a variables numericas
region_labels = encoder_Region.get_feature_names_out (["Region"]) # Dataframe

party_encoded = encoder_Party.fit_transform(df[["Previous_Answer"]]).toarray() # ahora transforma la respuesta anterior a numerico
party_labels = encoder_Party.get_feature_names_out(["Previous_Answer"])

# Se concatenan las columnas a la original, eliminando las que estaban
df = pd.concat ([df, pd.DataFrame(region_encoded, columns = region_labels)], axis = 1)
df = pd.concat ([df, pd.DataFrame(party_encoded, columns = party_labels)], axis = 1)

df.drop (columns = ["Region", "Previous_Answer"], inplace = True)

scaler = StandardScaler()
df["Income"] = scaler.fit_transform(df[["Income"]])

X = df.drop(columns=["Vote"])
y = df["Vote"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) #separa el modelo en 80% entrenamiento y 20% prueba


model = LogisticRegression(max_iter = 2000, class_weight = "balanced") #se crea el modelo para entrenarlo
model.fit (X_train, y_train) #Entrenamiento con los datos

y_pred = model.predict (X_test) #Predice en el conjunto de prueba

accuracy = accuracy_score (y_test, y_pred)
print (f"Model Accuracy: {accuracy: .2f}")

Conf_matrix = confusion_matrix(y_test, y_pred)
print ("\n Confusion Matrix : \n", Conf_matrix)

print ("\n Classification Report: \n", classification_report(y_test, y_pred))

def generate_plot ():
    plt.figure (figsize = (6, 4))
    plt.scatter (df ["Age"], df ["Vote"], c = df ["Vote"], cmap = "coolwarm", label = "Actual Data")
    plt.xlabel ("Age")
    plt.ylabel ("Vote (1 = yes, 0 = no)")
    plt.title ("Voting Distribution by Age")
    plt.legend ()

    img = io.BytesIO()
    plt.savefig(img, format = "png")
    img.seek(0)

    plot_url = base64.b64encode (img.getvalue()).decode()
    return plot_url