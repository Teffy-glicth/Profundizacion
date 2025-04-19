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
    "Age": [50, 58, 37, 52, 47, 26, 57, 38, 55, 34, 29, 19, 24, 73, 71, 48, 63, 68, 72, 62, 41, 54, 28, 73, 42, 23, 38, 60, 66, 59, 61, 53, 77, 77, 57, 73, 26, 63, 55, 27, 42, 40, 76, 27, 55, 23, 71, 19, 31, 67, 40, 49, 28, 30, 29, 37, 71, 77, 48, 20, 36, 26, 49, 44, 23, 21, 75, 73, 25, 59, 72, 21, 18, 32, 43, 23, 78, 18, 52, 77, 76, 56, 38, 68, 67, 22, 68, 54, 23, 29, 60, 64, 19, 23, 52, 34, 50, 49, 36, 53, 20, 31, 43, 26, 29, 65, 25, 77, 32],
    "Region": ["North", "West", "South", "East", "West", "South", "East", "West", "East", "South",
    "North", "South", "West", "North", "West", "East", "East", "North", "East", "West",
    "North", "West", "West", "South", "South", "South", "West", "South", "West", "East",
    "East", "South", "West", "South", "West", "North", "South", "South", "East", "West",
    "West", "North", "South", "East", "West", "West", "West", "South", "North", "West",
    "South", "North", "East", "South", "North", "West", "North", "East", "West", "East",
    "West", "North", "West", "North", "East", "North", "South", "East", "South", "North",
    "South", "North", "East", "West", "West", "East", "West", "South", "South", "North",
    "South", "South", "South", "South", "South", "South", "West", "West", "West", "West",
    "North", "West", "South", "South", "North", "North", "East", "East", "West", "West",
    "North", "West", "South", "East", "South", "West", "South", "South", "West"],
    "Previous_Answer": ["Centro Democrático", "Partido Liberal", "Partido Verde", "Pacto Histórico", "Partido Verde", "Partido Liberal", "Centro Democrático", "Partido Verde", "Partido Liberal", "Partido Verde", "Partido Liberal", "Pacto Histórico", "Partido Liberal", "Partido Verde", "Partido Liberal", "Partido Liberal", "Pacto Histórico", "Pacto Histórico", "Centro Democrático", "Partido Verde", "Partido Liberal", "Partido Liberal", "Partido Verde", "Partido Verde", "Centro Democrático", 
    "Centro Democrático", "Partido Verde", "Pacto Histórico", "Partido Verde", "Centro Democrático", "Partido Verde", "Partido Verde", "Centro Democrático", "Partido Verde", "Pacto Histórico", "Partido Verde", "Partido Liberal", "Pacto Histórico", "Pacto Histórico", "Partido Liberal", "Partido Verde", "Partido Verde", "Pacto Histórico", "Partido Liberal", "Centro Democrático", "Partido Liberal", "Centro Democrático", "Pacto Histórico", "Pacto Histórico", "Centro Democrático", "Partido Verde", 
    "Centro Democrático", "Partido Verde", "Pacto Histórico", "Centro Democrático", "Partido Verde", "Pacto Histórico", "Pacto Histórico", "Partido Liberal", "Pacto Histórico", "Partido Verde", "Pacto Histórico", "Pacto Histórico", "Pacto Histórico", "Partido Verde", "Partido Verde", "Centro Democrático", "Centro Democrático", "Centro Democrático", "Partido Liberal", "Pacto Histórico", "Partido Liberal", "Partido Verde", "Partido Liberal", "Partido Liberal", "Partido Liberal", "Partido Verde",
     "Centro Democrático", "Partido Liberal", "Partido Liberal", "Partido Verde", "Pacto Histórico",  "Partido Liberal", "Partido Verde", "Centro Democrático", "Centro Democrático", "Centro Democrático", "Partido Liberal", "Partido Liberal", "Partido Verde", "Partido Liberal", "Partido Liberal", "Centro Democrático", "Centro Democrático", "Partido Verde", "Centro Democrático", "Partido Verde", "Centro Democrático", "Partido Verde", "Partido Verde", "Partido Liberal", "Partido Verde", "Centro Democrático",
      "Centro Democrático", "Centro Democrático", "Partido Verde", "Pacto Histórico", "Partido Liberal", "Pacto Histórico"],
    "Income":[3253700, 6481210, 4123903, 5654798, 2323514, 6352658, 9489913, 5313573, 4653842, 7145406, 3448811, 9726592, 1853314, 3318645, 8698121, 8358549, 6744334, 2953291, 6625584, 2636955, 7849052, 1395743, 1972232, 4605298, 3356596, 4497598, 4057469, 4002161, 1715640, 7930760, 8229840, 3888667, 4629005, 3658065, 8377109, 8795521, 4007487, 3767682, 9784878, 6134444, 5527051, 4769488, 8629559, 8610246, 2802214, 7271758, 8052215, 5633802, 5640670, 3676837, 9152232, 5879757, 9124664, 7163048, 2801793, 
    9803672, 8330902, 9411900, 7893357, 1279152, 4438947, 5209866, 3115796, 2675349, 8878736, 6795904, 5827949, 6664643, 3471177, 5911051, 5152420, 7858405, 5109163, 2589922, 8157861, 6569669, 2005858, 4262114, 1347018, 2053743, 4229198, 9927064, 2614538, 6777331, 9675948, 3581538, 2314246, 2822841, 6629874, 4075927, 3677057, 2427285, 9877953, 8295719, 9763054, 5498392, 2631165, 5949936, 9683885, 2128589, 6641461, 3446502, 2671212, 1960667, 9634806, 6351417, 9619147, 8620225, 7254984],
    "Vote": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0]
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

def generate_plot():
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Age"], df["Vote"], c=df["Vote"], cmap="coolwarm", label="Actual Data", alpha=0.7)
    
    
    age_range = np.linspace(df["Age"].min(), df["Age"].max(), 300).reshape(-1, 1)
    
    X_temp = pd.DataFrame(age_range, columns=["Age"])
    
    for col in X.columns:
        if col != "Age":
            X_temp[col] = 0
    
    
    y_prob = model.predict_proba(X_temp)[:, 1]
    plt.plot(age_range, y_prob, 'g-', label='Logistic Curve', alpha=0.8)
    
    if 'X_test' in globals():
        try:
            plt.scatter(X_test["Age"], y_pred, c=y_pred, cmap="coolwarm", marker="x", s=100, label="Predicted", alpha=0.5, linewidths=1)
        except KeyError:
            age_test = X_test.iloc[:, 0]  
            plt.scatter(age_test, y_pred, c=y_pred, cmap="coolwarm", marker="x", s=100, label="Predicted", alpha=0.5)
    
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Vote (1 = yes, 0 = no)", fontsize=12)
    plt.title("Voting Distribution by Age\n(Actual vs Predicted)", fontsize=14, pad=20)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    img = io.BytesIO()
    plt.savefig(img, format="png", dpi=100, bbox_inches='tight')
    img.seek(0)
    plt.close()  
    
    return base64.b64encode(img.getvalue()).decode()