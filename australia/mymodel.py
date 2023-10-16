import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer # tri druhy skalovani
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split # pro rozdeleni  dat
from sklearn.model_selection import KFold # pouziti cross validace
from sklearn.pipeline import Pipeline


df = pd.read_csv("weatherAUS.csv")
print(df.head())
# ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
#        'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
#        'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
#        'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
#        'Temp3pm', 'RainToday', 'RainTomorrow']
df["avg_pressure"] = (df['Pressure3pm']+df['Pressure9am'])/2

df_r = df[['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'avg_pressure','RainToday', 'RainTomorrow']].copy()
df_r = df_r.dropna()
# dojde k odstraneni cca 20 000 hodnot z cca 145 000

# priprava dat pro uceni
y = preprocessing.LabelEncoder().fit_transform(df_r['RainTomorrow']) # prekoduj si na to zda prselo, ci ne
t = pd.to_datetime(df_r["Date"]).dt.day_of_year

# pripadne je mozne pouzit ordinal encoder na lokaci
X = pd.DataFrame()
X["sint"] = np.sin(2*3.14/365.25*t)
X["cost"] = np.cos(2*3.14/365.25*t)
X['MinTemp'] = df_r['MinTemp']
X['MaxTemp'] = df_r['MaxTemp']
X['Rainfall'] = df_r['Rainfall']
X['avg_pressure'] = df_r['avg_pressure']
X["RainToday"] = preprocessing.OrdinalEncoder().fit_transform(df_r['RainToday'].values.reshape(-1,1))
X['Location'] = preprocessing.OrdinalEncoder().fit_transform(df_r['Location'].values.reshape(-1,1)) # mozna onehot


#  nekde se mi tam objevily chzbejici hodnoty na konci/ tak to dropnu
X = X.dropna()

print("Vstupni data do modelu uceni")
#print(X.head(5))
X.to_csv("X.csv")

# validace modelu
seed = 42
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

pipe_sc_lr = Pipeline([('scaler', StandardScaler()),('classifier', LogisticRegression())])
pipe_mm_rf = Pipeline([('scaler', MinMaxScaler()),('classifier', RandomForestClassifier(n_estimators=200))])
pipe_mm_knn5 = Pipeline([('scaler', MinMaxScaler()),('classifier', KNeighborsClassifier(n_neighbors=5))])
pipe_mm_knn7 = Pipeline([('scaler', MinMaxScaler()),('classifier', KNeighborsClassifier(n_neighbors=7))])
pipe_sc_svc = Pipeline([('scaler', StandardScaler()),('classifier', SVC(gamma='auto'))])


pipes = {"pipe_sc_lr":pipe_sc_lr, "pipe_sc_knn7":pipe_mm_knn7,"pipe_mm_rf":pipe_mm_rf,"pipe_mm_knn5":pipe_mm_knn5,"pipe_sc_svc":pipe_sc_svc}
# pro kazdou rouru si budeme drzet vysledky
results = { pipe_name: [] for pipe_name in pipes.keys()}
# rozdelime si data na trenovaci, ktere budeme delit dale, a testovaci, na kterych pak ukazeme chovani
X_tr, X_test, y_tr, y_test = train_test_split(X,y, test_size=0.2,random_state=seed) # testovaci data zatim nepouzijeme
# trenovaci mnozinu budeme delit dale na 5 podmnozin
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X_tr, y_tr): # vraci dvojici poli testovacich a trenovacich indexu
  # rozdel si data na trenovaci a na data, na kterych bude ohodnocen klasifikator
  X_fold_tr = X_tr.values[train_index]
  y_fold_tr = y_tr[train_index]
  X_fold_test = X_tr.values[test_index]
  y_fold_test = y_tr[test_index]
  for k, pipe in pipes.items(): # pro kazdou pipe, delej
    pipe.fit(X_fold_tr, y_fold_tr) # nauc
    results[k].append(pipe.score(X_fold_test,y_fold_test)) # uloz si accuracy
#udelej si dataframe pro zhodnoceni
results = pd.DataFrame(data = results)
print(results) # tiskni za jednoltive foldy uspesnost klasifikatoru
print(results.mean()) # tiskni prumer