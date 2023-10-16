"""
1 READ
2 Data clean
  -location arrange
  -onehot
3 MINMAX MODELS
4 get baseline-done
?Goole dataset

bal = SMOTE()
x, y = bal.fit_resample(x, y) QUESTION ABOUT IT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer # tri druhy skalovani
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split # pro rozdeleni  dat
from sklearn.model_selection import KFold # pouziti cross validace
from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import catboost as cb
import xgboost as xgb
from sklearn.impute import KNNImputer

def remove_outliers(df,columns,n_std):
    i = 0
    for col in columns:
        i=+1
        print('Working on column:', i)
        
        mean = df[col].mean()
        sd = df[col].std()
        
        df = df[(df[col] <= mean+(n_std*sd))]
        
    return df

df = pd.read_csv("weatherAUS.csv")
print(df.head())
# ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
#        'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
#        'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
#        'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
#        'Temp3pm', 'RainToday', 'RainTomorrow']
print(df.shape)
imputer = KNNImputer(n_neighbors=1)

features_with_outliers = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'WindGustSpeed','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
for feature in features_with_outliers:
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    IQR = q3-q1
    lower_limit = q1 - (IQR*1.5)
    upper_limit = q3 + (IQR*1.5)
    df.loc[df[feature]<lower_limit,feature] = lower_limit
    df.loc[df[feature]>upper_limit,feature] = upper_limit


numerical_features = ['Sunshine','Cloud3pm',]#'Evaporation'
"""

numerical_features_with_null = [feature for feature in numerical_features if df[feature].isnull().sum()]
for feature in numerical_features_with_null:
    mean_value = df[feature].mean()
    df[feature].fillna(mean_value,inplace=True)
"""
print(df.shape)
df_r = df[['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Pressure3pm','Pressure9am',
'RainToday', 'RainTomorrow', 'Humidity3pm', 'Humidity9am','Temp3pm','WindGustSpeed','Temp9am','WindGustDir','Evaporation','Sunshine','Cloud3pm']].copy()
#df_r = df_r.dropna()
print(df.shape)

#remove_outliers(df, df_r, 3)
# dojde k odstraneni cca 20 000 hodnot z cca 145 000 ... 26 rn

# priprava dat pro uceni
y = preprocessing.LabelEncoder().fit_transform(df_r['RainTomorrow']) # prekoduj si na to zda prselo, ci ne
t = pd.to_datetime(df_r["Date"]).dt.day_of_year

# pripadne je mozne pouzit ordinal encoder na lokaci
X = pd.DataFrame()
X["sint"] = np.sin(2*3.14/365.25*t)
X["cost"] = np.cos(2*3.14/365.25*t)
X['MinTemp'] = df_r['MinTemp']
X['MaxTemp'] = df_r['MaxTemp']
X['Humidity3pm'] = df_r['Humidity3pm']
X['Humidity9am'] = df_r['Humidity9am']
X['WindGustSpeed'] = df_r['WindGustSpeed']
X['Temp3pm'] = df_r['Temp3pm']
#X['Temp9am'] = df_r['Temp9am']
X['Pressure3pm'] = df_r['Pressure3pm']
X['Pressure9am'] = df_r['Pressure9am']
#experimenta
#X['Evaporation'] = df_r['Evaporation']
print(X.shape)

X['Location'] = df_r['Location'] #onehot
one_hot = pd.get_dummies(X['Location'],drop_first=True)
X = X.drop('Location',axis = 1)
X = X.join(one_hot)

X['WindGustDir'] = df_r['WindGustDir'] #onehot
one_hot = pd.get_dummies(X['WindGustDir'],drop_first=True)
X = X.drop('WindGustDir',axis = 1)
X = X.join(one_hot)

X["RainToday"] = preprocessing.OrdinalEncoder().fit_transform(df_r['RainToday'].values.reshape(-1,1))
#  nekde se mi tam objevily chzbejici hodnoty na konci/ tak to dropnu
X['Cloud3pm'] = df_r['Cloud3pm']
X['Sunshine'] = df_r['Sunshine']
imputer.fit_transform(X,'Sunshine')
imputer.fit_transform(X,'Cloud3pm')
print("trans2done")
X = X.dropna()

print("Vstupni data do modelu uceni")
print(X.shape)
#print(X.head(5))
X.to_csv("X_v2.csv")

# validace modelu
seed = 42
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

pipe_sc_lr = Pipeline([('scaler', StandardScaler()),('classifier', LogisticRegression())])

kernel = 1.0 * RBF(1.0)

pipe_sc_gaus = Pipeline([('scaler', StandardScaler()),('classifier', GaussianNB())])

pipe_mm_rf = Pipeline([('scaler', MinMaxScaler()),('classifier', RandomForestClassifier(n_estimators=500))])
pipe_sc_rf = Pipeline([('scaler', StandardScaler()),('classifier', RandomForestClassifier(n_estimators=500,min_samples_split=3,min_samples_leaf=3))])

pipe_mm_knn5 = Pipeline([('scaler', MinMaxScaler()),('classifier', KNeighborsClassifier(n_neighbors=5))])

pipe_mm_knn7 = Pipeline([('scaler', MinMaxScaler()),('classifier', KNeighborsClassifier(n_neighbors=7))])

pipe_sc_svc = Pipeline([('scaler', StandardScaler()),('classifier', SVC(gamma='auto', cache_size=4000, C=0.8))])

pipe_sc_sgd = Pipeline([('scaler', StandardScaler()),('classifier', SGDClassifier(max_iter=1000, tol=1e-3))])

pipe_sc_dummy = Pipeline([('scaler', StandardScaler()),('classifier', DummyClassifier(strategy="most_frequent"))])

pipe_sc_dtree = Pipeline([('scaler', StandardScaler()),('classifier', tree.DecisionTreeClassifier())])
pipe_mm_dtree = Pipeline([('scaler', MinMaxScaler()),('classifier', tree.DecisionTreeClassifier())])

pipe_sc_cent = Pipeline([('scaler', StandardScaler()),('classifier', NearestCentroid())])
pipe_mm_cent = Pipeline([('scaler', MinMaxScaler()),('classifier', NearestCentroid())])

pipe_sc_ada = Pipeline([('scaler', StandardScaler()),('classifier', AdaBoostClassifier(n_estimators=5000))])#scaler is the same 0.811
pipe_mm_ada = Pipeline([('scaler', MinMaxScaler()),('classifier', AdaBoostClassifier(n_estimators=5000))])

pipe_sc_cat = Pipeline([('scaler', StandardScaler()),('classifier', cb.CatBoostClassifier(random_state=seed, verbose = 0))])
pipe_mm_cat = Pipeline([('scaler', MinMaxScaler()),('classifier', cb.CatBoostClassifier(random_state=seed, verbose = 0))])

pipe_sc_xcb = Pipeline([('scaler', StandardScaler()),('classifier', xgb.XGBClassifier(n_estimators=500, max_depth=16))])
pipe_mm_xcb = Pipeline([('scaler', MinMaxScaler()),('classifier', xgb.XGBClassifier(n_estimators=500, max_depth=16))])

clf1 = LogisticRegression(max_iter=200)
clf2 = RandomForestClassifier(n_estimators=500)
clf3 = xgb.XGBClassifier(n_estimators=500, max_depth=16)
clf4 = AdaBoostClassifier(n_estimators=1000)
clf5 = cb.CatBoostClassifier(random_state=seed, verbose = 0)
eclf = VotingClassifier(
estimators=[('lr', clf1), ('rf', clf2), ('xgbc', clf3), ('ada', clf4), ('cat', clf5), ],
voting='soft')

pipe_sc_voting = Pipeline([('scaler', StandardScaler()),('classifier', eclf)])


#"pipe_sc_gaus":pipe_sc_gaus
pipes = {"pipe_sc_lr":pipe_sc_lr,"pipe_mm_rf":pipe_mm_rf,"pipe_sc_cat":pipe_sc_cat}
#pipes = {"pipe_sc_voting":pipe_sc_voting,"pipe_mm_rf":pipe_mm_rf,"pipe_sc_ada":pipe_sc_ada,"pipe_sc_cat":pipe_sc_cat}
# pro kazdou rouru si budeme drzet vysledky
results = { pipe_name: [] for pipe_name in pipes.keys()}
# rozdelime si data na trenovaci, ktere budeme delit dale, a testovaci, na kterych pak ukazeme chovani
X_tr, X_test, y_tr, y_test = train_test_split(X,y, test_size=0.2,random_state=seed) # testovaci data zatim nepouzijeme
# trenovaci mnozinu budeme delit dale na 5 podmnozin
kf = KFold(n_splits=5, shuffle=True)
print("start")
for train_index, test_index in kf.split(X_tr, y_tr): # vraci dvojici poli testovacich a trenovacich indexu
  # rozdel si data na trenovaci a na data, na kterych bude ohodnocen klasifikator
  X_fold_tr = X_tr.values[train_index]
  y_fold_tr = y_tr[train_index]
  X_fold_test = X_tr.values[test_index]
  y_fold_test = y_tr[test_index]
  for k, pipe in pipes.items(): # pro kazdou pipe, delej
    pipe.fit(X_fold_tr, y_fold_tr) # nauc
    results[k].append(pipe.score(X_fold_test,y_fold_test)) # uloz si accuracy
    print(train_index,k)
#udelej si dataframe pro zhodnoceni
print("progress6")
results = pd.DataFrame(data = results)

print(results) # tiskni za jednoltive foldy uspesnost klasifikatoru
print(results.mean()) # tiskni prumer



