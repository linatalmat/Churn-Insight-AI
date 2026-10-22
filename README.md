
# Churn-Insight-AI

## Présentation

Churn-Insight-AI est un projet de Machine Learning visant à prédire le risque de résiliation des clients (*customer churn*).

L'objectif est d'identifier les clients susceptibles de quitter une entreprise afin de mieux comprendre leur comportement et d'aider à améliorer leur fidélisation.

Le projet comprend l'analyse et le nettoyage des données, la préparation des variables, l'entraînement d'un modèle de Machine Learning et le développement d'une application interactive avec Streamlit.

## Objectifs du projet

- Analyser les données clients.
- Identifier les facteurs associés à la résiliation.
- Préparer les données pour le Machine Learning.
- Entraîner un modèle de prédiction du churn.
- Évaluer les performances du modèle.
- Développer une application permettant de prédire le risque de résiliation d'un client.

## Technologies utilisées

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook
- Git et GitHub

## Structure du projet

```text
Churn-Insight-AI/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── customer_churn.csv
│   └── processed/
│       └── costumer_churn_clean.csv
│
├── docs/
│   └── presentation_projet.md
│
├── notebooks/
│   ├── 01_analyse_exploratoire.ipynb
│   ├── 02_nettoyage_donnees.ipynb
│   ├── 03_preparation_features.ipynb
│   ├── 04_machine_learning.ipynb
│   └── models/
│       └── churn_model.pkl
│
└── README.md
```

## Étapes du projet

### 1. Analyse exploratoire

Exploration du jeu de données pour comprendre les caractéristiques des clients et identifier les tendances liées à la résiliation.

### 2. Nettoyage des données

Traitement des valeurs manquantes, vérification des données et préparation du jeu de données.

### 3. Préparation des variables

Transformation et sélection des variables nécessaires à l'entraînement du modèle.

### 4. Modélisation

Entraînement d'un modèle de classification permettant de prédire si un client est susceptible de résilier son abonnement.

### 5. Évaluation

Évaluation du modèle à l'aide de plusieurs métriques :

- Matrice de confusion
- Precision
- Recall
- F1-score

### 6. Application Streamlit

Développement d'une interface interactive permettant d'effectuer des prédictions de résiliation à partir des données d'un client.

## Installation et lancement

Cloner le dépôt GitHub :

```bash
git clone https://github.com/linatalmat/Churn-Insight-AI.git
```

Accéder au dossier :

```bash
cd Churn-Insight-AI
```

Activer l'environnement virtuel sous Windows PowerShell :

```powershell
.\.venv312\Scripts\Activate.ps1
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer l'application :

```bash
streamlit run app/app.py
```

## Dépôt GitHub

[Consulter le projet Churn-Insight-AI](https://github.com/linatalmat/Churn-Insight-AI)
