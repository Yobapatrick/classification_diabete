from pydantic import BaseModel  #pour la validation des données
import numpy as np
import pandas as pd  
import joblib  
from flask import Flask, request, jsonify  

#chargement du fichier contenant le modèle de forêt aléatoire depuis le disque
modele = joblib.load('Modele/random_forest_model.pkl')

#Définition du schéma des données d'entrée avecPydantic
class DonneesEntree(BaseModel):
    Pregnancies: float  
    Glucose: float  
    BloodPressure: float  
    SkinThickness: float  
    Insulin: float  
    BMI: float  
    DiabetesPedigreeFunction: float  
    Age: float  

#Création de l'instance de l'application flask
app = Flask(__name__)

#Définition de la route racine qui retourne un message de bienvenue
@app.route("/", methods=["GET"])
def accueil():
    """ Endpoint racine qui fournit un message de bienvenue. """
    return jsonify({"message": "Bienvenue sur l'API de prédiction pour le diagnostic du diabète"})

#Définition de la route pour les prédictions de diabète
@app.route("/predire", methods=["POST"])
def predire():
    """
    Endpoint pour les prédictions en utilisant le modèle chargé.
    Les données d'entrée sont validées et transformées en DataFrame pour le traitement par le modèle.
    """
    if not request.json:
        return jsonify({"erreur": "Aucun JSON fourni"}), 400
    
    
    try:
        # Extraction et validation des données via  Pydantic
        donnees = DonneesEntree(**request.json)
        donnees_df = pd.DataFrame([donnees.dict()])  

        # Utilisation du modèle pour prédire et obtenir les probabilités
        predictions = modele.predict(donnees_df)
        probabilities = modele.predict_proba(donnees_df)[:, 1]  # Probabilité de la classe positive (diabète)

        # Compilation des résultats dans un dictionnaire
        resultats = donnees.dict()
        resultats['prediction'] = int(predictions[0])
        resultats['probabilite_diabete'] = probabilities[0]

        # Renvoie les résultats sous forme de JSON
        return jsonify({"resultats": resultats})
    except Exception as e:
        return jsonify({"erreur": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=8000)  # Lancement de l'application sur le port 8000 avec le mode debug activé