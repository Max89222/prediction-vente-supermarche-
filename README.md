# Création d'un modèle de prédiction des ventes dans différents supermarchés

**Auteur : Max89222**

---

Ce projet vise à créer un modèle capable de prédire le nombre de ventes par catégorie dans 52 supermarchés en Équateur, en utilisant un modèle de régression.

## 1) Structure et utilité des fichiers

- `holidays_events.csv` : contient des informations sur les jours fériés ainsi que les fêtes locales, régionales et nationales en Équateur.  
- `oil.csv` : évolution du prix du pétrole.  
- `stores.csv` : informations sur les différents magasins étudiés (localisation, type de magasin, etc.).  
- `transactions.csv` : historique des transactions des différents supermarchés.  
- `train.csv` : historique des ventes par jour et par catégorie de produits.  
- `test.csv` : même structure que `train.csv` mais sans les colonnes de ventes (ce sont elles qu'on cherche à prédire).  
- `sample_submission.csv` : fichier servant de modèle pour formater nos prédictions sur `test.csv`.  
- `prediction_kaggle.csv` : fichier final à soumettre sur Kaggle, contenant nos prédictions.  

Ce dataset provient du site Kaggle : [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

## 2) Fonctionnement du modèle

J’ai d’abord divisé le dataset en un ensemble d'entraînement (`train_set`) et un ensemble de test (`test_set`) afin d'entraîner un premier modèle (`DecisionTreeRegressor`) chargé de prédire le nombre de transactions dans `test.csv`.  
Ces prédictions ont ensuite été utilisées comme variable d'entrée pour un second modèle (`RandomForestRegressor`) destiné à prédire le nombre de ventes par catégorie.

## 3) Technologies utilisées

- `pandas` : manipulation et nettoyage des données  
- `scikit-learn` : création et évaluation des modèles de machine learning  
- `matplotlib` : visualisation des données  
- `numpy` : opérations numériques  
- `joblib` : sauvegarde et chargement des modèles

Le modèle final utilisé est un **RandomForestRegressor**, qui s’est avéré le plus performant dans ce problème de régression.

## 4) Résultats et métriques

Score obtenu sur Kaggle (Root Mean Squared Logarithmic Error) : **0.56005**

## 5) Installation

1. Installer Git (si ce n’est pas déjà fait) :
   
`brew install git`

2. Cloner le dépôt :

`git clone <clé_ssh>`
`cd <nom_du_dossier>`

3. Installer les dépendances :

`pip3 install pandas scikit-learn matplotlib numpy joblib`

4. Entraîner le modèle :

Ouvrir main.py dans un éditeur de code et l'exécuter.
⚠️ Attention : l'entraînement peut être long à cause du volume de données.
A la fin de l'entraînement, un fichier `model_prediction_transactions.pkl` et `model_prediction_ventes.pkl` seront générés. 
Ceux ci vous permettront par la suite de pouvoir utiliser le modèle (faire des prédictions) sans avoir à entraîner le modèle une seconde fois.

Ce projet a une visée éducative et ne prétend pas à une application réelle. L'objectif est d’évaluer la qualité du code et la logique algorithmique.

## 6) Idées d'amélioration et contributions
De nombreux points restent à améliorer. N'hésitez pas à contribuer à ce projet ou à proposer des idées d'optimisation !

