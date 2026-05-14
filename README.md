<div align="center">

# 🩺 Diabetes Classification — End-to-End ML Service

**Un classifieur binaire de risque de diabète, du notebook d'entraînement à l'API déployée en production.**

*De l'analyse exploratoire au endpoint HTTP conteneurisé — un projet pensé comme un produit, pas comme un exercice.*

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-Validation-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Cloud Run](https://img.shields.io/badge/Google%20Cloud%20Run-Deployed-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/run)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Aperçu](#-aperçu) · [Architecture](#-architecture) · [Pipeline ML](#-pipeline-machine-learning) · [Résultats](#-résultats--métriques) · [Installation](#-installation--démarrage) · [API](#-utilisation-de-lapi) · [Roadmap](#-roadmap)

</div>

---

## 📌 Aperçu

Ce projet répond à une question clinique simple à formuler, difficile à modéliser :

> **À partir de 8 mesures diagnostiques de routine, peut-on estimer le risque qu'une patiente soit diabétique ?**

Plutôt que de s'arrêter à un notebook, le projet va jusqu'au bout de la chaîne de valeur d'un modèle ML : il **entraîne**, **évalue**, **sérialise**, **expose via une API**, **conteneurise** et **déploie** le modèle, avec une interface Streamlit pour les prédictions par lot.

L'objectif n'est pas seulement d'obtenir un score, mais de démontrer une **méthodologie d'ingénieur ML** : reproductibilité, séparation des responsabilités, validation des entrées, et — point central de ce projet — **détection et correction d'une fuite de données** qui faussait initialement les performances.

### Pourquoi ce projet compte

- 🎯 **End-to-end, pas un jouet** — la majorité des projets de classification s'arrêtent au `.fit()`. Celui-ci produit un service interrogeable.
- 🔍 **Rigueur méthodologique** — l'histoire de ce projet inclut la découverte d'une fuite de données et sa correction. C'est exactement le type de problème qui coûte cher en production.
- 🧱 **Pratiques software** — schémas de validation typés, conteneur reproductible, tests, structure modulaire.
- 🩺 **Domaine à enjeu** — le coût d'un faux négatif (diabétique classé sain) n'est pas le coût d'un faux positif. Le projet en tient compte dans le choix des métriques.

---

## 🎬 Démonstration

> 📍 *Remplacer par un GIF de l'interface Streamlit (`assets/demo.gif`) et/ou un lien vers la démo live.*

| Interface Streamlit | Réponse de l'API |
|---|---|
| ![Streamlit UI](assets/streamlit_demo.png) | ![API response](assets/api_response.png) |

**Démo live :** `https://<votre-service>.run.app` *(Google Cloud Run)*

---

## 🧩 Problématique & motivation

Le diabète de type 2 est largement sous-diagnostiqué : une détection précoce change le pronostic. Les outils de dépistage assistés par ML ne remplacent pas un médecin, mais peuvent **prioriser** les patientes à examiner.

Le défi technique réel ici n'est pas l'algorithme — un Random Forest sur 8 variables est trivial à entraîner. Le défi est de construire autour de ce modèle **tout ce qui le rend utilisable et fiable** : une interface stable, une validation stricte des entrées, un déploiement reproductible, et surtout une **évaluation honnête** qui ne se laisse pas tromper par des artefacts de données.

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Dataset       │     │   Notebook       │     │  random_forest_     │
│  diabetes.csv   │ ──▶ │  EDA + Training  │ ──▶ │  model.pkl          │
│  (Pima, 768)    │     │  + Évaluation    │     │  (modèle sérialisé) │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                            │
                                                            ▼
┌──────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Streamlit UI    │ ──▶ │   Flask API      │ ──▶ │   Random Forest     │
│  (batch CSV)     │     │  /predire (POST) │     │   .predict_proba()  │
│                  │ ◀── │  Pydantic schema │ ◀── │                     │
└──────────────────┘     └────────┬─────────┘     └─────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │  Docker → Google Cloud   │
                    │  Run (gunicorn, $PORT)   │
                    └──────────────────────────┘
```

**Flux de données :** le notebook entraîne et exporte le modèle → l'API Flask le charge au démarrage et expose un endpoint `/predire` → Pydantic valide chaque requête entrante → le modèle renvoie une classe **et** une probabilité → l'interface Streamlit consomme l'API pour des prédictions par lot.

---

## 🛠️ Stack technique

| Couche | Technologies | Rôle |
|---|---|---|
| **Données & ML** | pandas, NumPy, scikit-learn | Manipulation, EDA, entraînement du Random Forest |
| **Visualisation** | matplotlib, seaborn | Histogrammes, boxplots, pairplot, heatmap, courbe ROC, feature importance |
| **Sérialisation** | joblib | Persistance du modèle entraîné (`.pkl`) |
| **API** | Flask, Pydantic | Service d'inférence HTTP + validation typée des entrées |
| **Interface** | Streamlit | Front pour prédiction par lot via upload CSV |
| **Serveur** | Gunicorn | Serveur WSGI de production (1 worker, 8 threads) |
| **Conteneurisation** | Docker | Image reproductible basée sur `python:3.9-slim` |
| **Déploiement** | Google Cloud Run | Hébergement serverless, scaling automatique |

---

## 🔬 Pipeline Machine Learning

### 1. Le dataset

Le projet s'appuie sur le **Pima Indians Diabetes Dataset** — un jeu de référence en classification binaire médicale, composé de **768 observations** de patientes (femmes d'origine Pima, ≥ 21 ans).

| Variable | Description | Unité |
|---|---|---|
| `Pregnancies` | Nombre de grossesses | entier |
| `Glucose` | Glycémie plasmatique à jeun | mg/dL |
| `BloodPressure` | Pression artérielle diastolique | mm Hg |
| `SkinThickness` | Épaisseur du pli cutané tricipital | mm |
| `Insulin` | Insuline sérique à 2 h | mu U/ml |
| `BMI` | Indice de masse corporelle | kg/m² |
| `DiabetesPedigreeFunction` | Score de prédisposition héréditaire | ratio |
| `Age` | Âge | années |
| **`Outcome`** | **Cible — 1 = diabétique, 0 = non diabétique** | binaire |

**Répartition des classes :** ~65 % non-diabétiques / ~35 % diabétiques — un déséquilibre modéré qui justifie de regarder le **recall** et non seulement l'accuracy.

### 2. ⚠️ Intégrité des données — détection d'une fuite

> **C'est le point méthodologique central du projet.**

La première version du fichier d'entraînement contenait **2 769 lignes** — mais seulement **768 réellement uniques**. Le dataset Pima avait été dupliqué ~3,6×, et une ligne d'en-tête s'était glissée dans les données.

**Conséquence :** comme `train_test_split` répartit aléatoirement les lignes, des **copies identiques se retrouvaient à la fois dans l'ensemble d'entraînement et de test**. Le modèle « voyait » donc ses réponses de test pendant l'entraînement → une **fuite de données** classique.

Symptôme révélateur : une accuracy de **99,8 %** et un AUC de **0,999** sur le test. Des scores « trop beaux » sont presque toujours le signe d'un bug, pas d'un génie.

**Correction appliquée :**
1. Suppression de la ligne d'en-tête parasite
2. **Déduplication** → retour aux 768 observations uniques
3. Ré-évaluation honnête sur des splits propres

> 💡 *Leçon retenue : un score doit toujours être interrogé avant d'être célébré. La capacité à débusquer une fuite de données vaut plus, en production, que quelques points d'accuracy.*

### 3. Analyse exploratoire (EDA)

Le notebook produit une exploration visuelle systématique :
- **Distributions** — histogrammes des 8 variables explicatives
- **Détection d'anomalies** — boxplots révélant outliers et valeurs biologiquement impossibles (glycémie, IMC ou insuline à 0)
- **Relations** — `pairplot` coloré par `Outcome` pour visualiser la séparabilité des classes
- **Corrélations** — heatmap des variables numériques

### 4. Prétraitement

- Renommage des colonnes en français pour aligner le schéma métier (`Glucose` → `ConcentrationGlucose`, etc.)
- Identification des `0` biologiquement impossibles dans les colonnes cliniques comme valeurs manquantes implicites *(piste d'amélioration : imputation par la médiane — voir [Roadmap](#-roadmap))*
- Séparation features / cible, split **80 % train / 20 % test** (`random_state=42` pour la reproductibilité)

### 5. Modélisation — Random Forest

Le choix du **Random Forest** est délibéré :
- robuste aux outliers et aux échelles hétérogènes des variables (pas de normalisation requise) ;
- fournit nativement une **importance des variables** — précieuse pour l'interprétabilité clinique ;
- expose un **score Out-Of-Bag** qui donne une estimation de généralisation sans set de validation dédié.

| Hyperparamètre | Valeur | Justification |
|---|---|---|
| `n_estimators` | 100 | Compromis variance / coût de calcul |
| `max_features` | `√n_features` ≈ 2 | Décorrélation des arbres (règle classique en classification) |
| `oob_score` | `True` | Estimation de généralisation gratuite |
| `random_state` | 42 | Reproductibilité |

### 6. Évaluation

L'évaluation combine **accuracy**, **AUC** et **recall** — ce dernier étant prioritaire dans un contexte médical, car **un faux négatif (diabétique manqué) est plus grave qu'un faux positif**. Une **validation croisée 5-fold** complète l'évaluation pour vérifier la stabilité du modèle.

---

## 📊 Résultats & métriques

> ⚠️ Les chiffres ci-dessous sont les performances **attendues sur le dataset propre et dédupliqué**. Les scors de la première version (~99 %) étaient un artefact de fuite de données — voir la section [Intégrité des données](#2-️-intégrité-des-données--détection-dune-fuite). **À actualiser avec vos résultats après ré-entraînement.**

| Métrique | Train | Test | Lecture |
|---|---|---|---|
| **Accuracy** | ~0.85 | **~0.77** | Référence honnête pour le Pima dataset |
| **AUC** | ~0.90 | **~0.83** | Bonne capacité de discrimination |
| **Recall (classe diabétique)** | — | **à mesurer** | Métrique prioritaire — minimiser les faux négatifs |
| **CV 5-fold (accuracy)** | — | **~0.76 ± écart** | Confirme la stabilité, sans fuite |

### Importance des variables

L'analyse d'importance issue du Random Forest est cliniquement cohérente :

| Rang | Variable | Importance |
|---|---|---|
| 1 | **ConcentrationGlucose** | ~0.24 |
| 2 | IndiceMasse (IMC) | ~0.18 |
| 3 | FonctionPedigree | ~0.14 |
| 4 | Age | ~0.13 |
| 5 | PressionArterielle | ~0.10 |
| 6 | TauxInsuline | ~0.07 |
| 7 | EpaisseurPli | ~0.07 |
| 8 | Grossesses | ~0.07 |

> La **glycémie** domine — c'est le marqueur diagnostique du diabète, ce qui valide la cohérence du modèle. À l'inverse, le nombre de grossesses est le moins prédictif.

### Visualisations

| Courbe ROC | Importance des variables |
|---|---|
| ![ROC](assets/roc_curve.png) | ![Feature importance](assets/feature_importance.png) |

> 📍 *Exporter ces figures depuis le notebook vers `assets/`.*

---

## 📁 Structure du projet

```
diabetes-classification/
├── README.md
├── LICENSE
├── requirements.txt          # dépendances épinglées
├── Dockerfile                # image de production (python:3.9-slim + gunicorn)
├── .dockerignore
│
├── data/
│   └── raw/diabetes.csv      # Pima Indians Diabetes (768 lignes, dédupliqué)
│
├── notebooks/
│   └── 01_eda_and_modeling.ipynb   # EDA → entraînement → évaluation → export
│
├── api/
│   └── app.py                # API Flask + schéma Pydantic
│
├── app/
│   └── streamlit_app.py      # interface de prédiction par lot
│
├── models/
│   └── random_forest_model.pkl
│
├── tests/
│   └── test_api.py           # tests d'intégration de l'API
│

```

> 🔧 **Note d'évolution :** la structure historique du repo (`Modele/`, `Test/`, `api.py` à la racine) est en cours de refactorisation vers l'arborescence ci-dessus pour aligner le projet sur les standards open-source. Voir la [Roadmap](#-roadmap).

---

## 🚀 Installation & démarrage

### Prérequis

- Python 3.9+
- Docker *(optionnel, pour le déploiement conteneurisé)*

### Installation locale

```bash
# 1. Cloner le dépôt
git clone https://github.com/Yobapatrick/classification_diabete.git
cd classification_diabete

# 2. Environnement virtuel
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate

# 3. Dépendances
pip install -r requirements.txt
```

### Lancer l'API en local

```bash
python api/app.py
# API disponible sur http://127.0.0.1:8000
```

### Lancer l'interface Streamlit

```bash
streamlit run app/streamlit_app.py
```

### Déploiement conteneurisé

```bash
# Build
docker build -t diabetes-api .

# Run
docker run -p 8000:8000 -e PORT=8000 diabetes-api
```

> ⚙️ **À vérifier avant déploiement :** le `CMD` du Dockerfile doit pointer vers le bon module (`app:app` selon la structure), et le dossier `models/` ne doit **pas** être exclu par `.dockerignore` — sinon le `.pkl` est absent de l'image.

---

## 🔌 Utilisation de l'API

### `GET /` — Health check

```bash
curl http://127.0.0.1:8000/
```

```json
{ "message": "Bienvenue sur l'API de prédiction pour le diagnostic du diabète" }
```

### `POST /predire` — Prédiction

Le corps de la requête est **validé par Pydantic** : un champ manquant ou un type incorrect renvoie une erreur 400 explicite.

```bash
curl -X POST http://127.0.0.1:8000/predire \
  -H "Content-Type: application/json" \
  -d '{
    "Grossesses": 2,
    "ConcentrationGlucose": 138,
    "PressionArterielle": 62,
    "EpaisseurPli": 35,
    "TauxInsuline": 0,
    "IndiceMasse": 33.6,
    "FonctionPedigree": 0.127,
    "Age": 47
  }'
```

**Réponse :**

```json
{
  "resultats": {
    "Grossesses": 2,
    "ConcentrationGlucose": 138,
    "PressionArterielle": 62,
    "EpaisseurPli": 35,
    "TauxInsuline": 0,
    "IndiceMasse": 33.6,
    "FonctionPedigree": 0.127,
    "Age": 47,
    "prediction": 1,
    "probabilite_diabete": 0.71
  }
}
```

| Champ | Type | Description |
|---|---|---|
| `prediction` | `int` | `1` = diabétique prédit, `0` = non diabétique |
| `probabilite_diabete` | `float` | Probabilité de la classe positive ∈ [0, 1] |

> 📋 **Convention de schéma :** l'API utilise des noms de champs **français** (`Grossesses`, `ConcentrationGlucose`…). Tout client — y compris l'interface Streamlit — doit respecter ce contrat.

---

## 🧪 Tests

```bash
pytest tests/ -v
```

Les tests couvrent :
- le démarrage de l'API et le endpoint de santé ;
- une prédiction valide (payload complet) ;
- la gestion des erreurs (payload incomplet, types invalides) ;
- la cohérence du format de réponse.

---

## 🏭 Considérations de production

Ce projet est un **prototype déployable**, pas un système clinique certifié. Voici ce qui le sépare d'un service réellement production-grade — et qui constitue la feuille de route technique :

| Aspect | État actuel | Cible production |
|---|---|---|
| **Reproductibilité** | `requirements.txt` à épingler | Versions figées + lockfile |
| **Validation d'entrée** | ✅ Pydantic | + bornes métier (glycémie > 0, âge réaliste) |
| **Observabilité** | ❌ Aucune | Logging structuré, métriques, traçabilité des prédictions |
| **Tests** | Tests d'intégration de base | + tests unitaires, couverture, tests de charge |
| **CI/CD** | ❌ Manuel | GitHub Actions : lint → tests → build → déploiement |
| **Monitoring du modèle** | ❌ Aucun | Détection de *data drift* et de dégradation de performance |
| **Gestion du modèle** | `.pkl` versionné dans Git | Registry de modèles (MLflow / artefact versionné) |
| **Sécurité** | ❌ Endpoint ouvert | Authentification, rate limiting |

---

## ⚠️ Limitations

- **Dataset** — 768 observations, population spécifique (femmes Pima ≥ 21 ans) : le modèle **ne généralise pas** à d'autres populations.
- **Pas d'usage clinique** — outil pédagogique et de démonstration ; aucune valeur diagnostique réelle.
- **Valeurs manquantes implicites** — les `0` biologiquement impossibles ne sont pas encore imputés (voir Roadmap).
- **Pas de tuning d'hyperparamètres** — le modèle utilise des valeurs par défaut raisonnables, sans `GridSearchCV`.
- **Modèle unique** — pas de comparaison avec d'autres familles d'algorithmes (régression logistique, gradient boosting).

---

## 🗺️ Roadmap

### Court terme — fiabilité
- [ ] Refactoriser l'arborescence vers la structure modulaire (`api/`, `app/`, `src/`, `tests/`)
- [ ] Épingler les versions dans `requirements.txt`
- [ ] Imputation des `0` biologiquement impossibles (médiane par classe)
- [ ] Aligner le schéma entre l'API et tous les clients
- [ ] Corriger le `CMD` du Dockerfile et l'inclusion du dossier `models/`

### Moyen terme — qualité ML
- [ ] `GridSearchCV` / `RandomizedSearchCV` pour le tuning d'hyperparamètres
- [ ] Comparaison multi-modèles (Logistic Regression, XGBoost, LightGBM)
- [ ] Gestion explicite du déséquilibre de classes (`class_weight`, SMOTE)
- [ ] Calibration des probabilités + analyse du seuil de décision orientée recall
- [ ] Suite de tests complète (unitaires + intégration + couverture)

### Long terme — MLOps
- [ ] Pipeline CI/CD via GitHub Actions
- [ ] Suivi d'expériences avec MLflow
- [ ] Logging structuré et monitoring du *data drift*
- [ ] `docker-compose` orchestrant API + Streamlit
- [ ] Endpoint d'explicabilité (valeurs SHAP par prédiction)

---

## 🎓 Lessons Learned

- **Un score trop beau est un bug, pas un trophée.** La fuite de données détectée dans ce projet (99,8 % d'accuracy) est le rappel le plus utile : toujours interroger un résultat avant de le présenter.
- **Le déploiement révèle les vrais bugs.** Le passage du notebook à l'API a fait apparaître des incohérences de schéma et de configuration invisibles tant que le code restait dans une cellule.
- **La validation typée des entrées n'est pas optionnelle.** Pydantic transforme un endpoint fragile en contrat explicite — un client qui se trompe reçoit une erreur claire, pas une réponse silencieusement fausse.
- **Le choix de la métrique est une décision métier.** En contexte médical, optimiser l'accuracy au détriment du recall revient à manquer des patients malades. La métrique se choisit avant le modèle.

---

## 📚 Références

- **Dataset** — Pima Indians Diabetes Database, originellement du *National Institute of Diabetes and Digestive and Kidney Diseases* (disponible sur Kaggle / UCI ML Repository).
- **scikit-learn** — [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- **Évaluation** — Saito & Rehmsmeier, *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets* (2015)
- **Déploiement** — [Google Cloud Run — Deploying Python services](https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service)

---

## 🤝 Contributions

Les retours et contributions sont les bienvenus. Pour proposer une amélioration :

1. Forkez le dépôt
2. Créez une branche (`git checkout -b feature/amelioration`)
3. Commitez vos changements (`git commit -m 'Ajout: ...'`)
4. Poussez la branche et ouvrez une Pull Request

Les pistes de la [Roadmap](#-roadmap) sont d'excellents points de départ.

---

## 📄 Licence

Distribué sous licence MIT. Voir [`LICENSE`](LICENSE) pour les détails.

---

<div align="center">

**Réalisé par [Yoba Patrick](https://github.com/Yobapatrick)**

Si ce projet vous a été utile, n'hésitez pas à laisser une ⭐.

</div>
