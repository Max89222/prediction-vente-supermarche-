import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_log_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.feature_selection import f_regression, chi2,  SelectKBest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import joblib


pd.set_option('display.max_rows', 100)  
pd.set_option('display.max_columns', 100)  

# ouverture fichiers

train_set = pd.read_csv('train.csv')
jour_fer = pd.read_csv('holidays_events.csv')
info_mag = pd.read_csv('stores.csv')
oil_evo = pd.read_csv('oil.csv')
X_pred = pd.read_csv('test.csv')
transac = pd.read_csv('transactions.csv')
fichier_predictions = pd.read_csv('sample_submission.csv')


jour_fer = jour_fer.drop(['description', 'type', 'transferred'], axis=1)

train_set = train_set.drop('id', axis=1)

# conversion colonnes date ----> datetime

jour_fer['date'] = pd.to_datetime(jour_fer['date'])
oil_evo['date'] = pd.to_datetime(oil_evo['date'])
train_set['date'] = pd.to_datetime(train_set['date'])
X_pred['date'] = pd.to_datetime(X_pred['date'])
transac['date'] = pd.to_datetime(transac['date'])


def trans_date(train_set, trans=True): 

    train_set = pd.merge(train_set, info_mag, on='store_nbr', how='inner')
    train_set = pd.merge(train_set, jour_fer, on='date', how='left')
    train_set = pd.merge(train_set, oil_evo, on='date', how='left')
    if trans:
        train_set = pd.merge(train_set, transac, on=['date', 'store_nbr'], how='inner')

    
    train_set['fete_locale'] = (train_set['locale'] == 'Local') & (train_set['locale_name'] == train_set['city'])
    train_set['fete_nationale'] = train_set['locale'] == 'National'
    train_set['fete_locale'] = train_set['fete_locale'].astype(int)
    train_set['fete_nationale'] =  train_set['fete_nationale'].astype(int)


    train_set = train_set.drop(['locale', 'locale_name'], axis=1)


    train_set['dcoilwtico'] = train_set['dcoilwtico'].interpolate(method='linear')
    train_set['dcoilwtico'] = train_set['dcoilwtico'].fillna(method='bfill')  
    train_set['dcoilwtico'] = train_set['dcoilwtico'].fillna(method='ffill')  

    train_set['jour_semaine'] = train_set['date'].dt.dayofweek
    train_set['mois'] = train_set['date'].dt.month
    train_set['annee'] = train_set['date'].dt.year
    train_set['jour'] = train_set['date'].dt.day
    train_set['noel'] = ((train_set['jour']== 24) | (train_set['jour']== 25) | (train_set['jour']== 23)) & (train_set['mois'] == 12)
    train_set['noel'] = train_set['noel'].astype(int)
    train_set = train_set.drop('date', axis=1)

    return train_set


train_set = trans_date(train_set, trans=True)

object_col = ('family', 'city', 'state', 'type', 'store_nbr', 'cluster', 'jour_semaine', 'fete_nationale', 'fete_locale', 'mois', 'jour', 'noel')
numerical_col = ('onpromotion', 'dcoilwtico')

X = train_set.drop('sales', axis=1)
y = train_set['sales']


#plt.plot(transaction_moyenne_par_jour.index, transaction_moyenne_par_jour, label='nbr transac')
#plt.plot(prix_petrol_moyen_par_jour.index, prix_petrol_moyen_par_jour*50, label='prix pétrol')
#plt.legend()
#print(train_set[['transactions', 'dcoilwtico']].corr())
#plt.show()


categorical_pipeline = make_pipeline(OrdinalEncoder())
numerical_pipeline = make_pipeline(StandardScaler())


preprocessor = ColumnTransformer([
    ('cat', categorical_pipeline, object_col), 
    ('num', numerical_pipeline, numerical_col)])

model_trans = Pipeline([
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor(random_state=0)) 
     ])

model_sales = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_jobs=-1, random_state=0, n_estimators=10, max_features=0.5, max_depth=25))
     ])



X_trans = train_set.drop('transactions', axis=1)
y_trans = train_set['transactions']
print(X_trans.columns)

print('début entraînement modèle transactions')
model_trans.fit(X_trans, y_trans)
print('fin entraînement entraînement modèle transactions')



X_final = trans_date(X_pred, trans=False)


X_final['transactions'] = model_trans.predict(X_final)

X_sales = train_set.drop('sales', axis=1)
y_sales = train_set['sales']


print('début entraînement prédiction sales')
model_sales.fit(X_sales, y_sales)
selector_sales = model_sales.named_steps['preprocessor']
print('fin entraînement prédiction sales')

final_pred = model_sales.predict(X_final)

fichier_predictions['sales'] = final_pred

fichier_predictions.to_csv('prediction_kaggle.csv', index=False)


joblib.dump(model_sales, 'model_prediction_ventes')
joblib.dump(model_trans, 'model_prediction_transactions')
