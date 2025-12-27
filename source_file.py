import pickle

import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

#parameters
C = 1.0
n_splits = 5
output_file = f'model_{C}.bin'

#data preparation

df = pd.read_csv('C:\\Users\\PRECIOUS\\Downloads\\course_lead_scoring.csv')

numerical_cols = df.select_dtypes(include=np.number).columns

df[numerical_cols] = df[numerical_cols].fillna(0.0)

categorical_cols = df.select_dtypes(include='object').columns

df[categorical_cols] = df[categorical_cols].fillna('NA')

#spliting

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


categorical = ['lead_source' , 'industry' , 'employment_status' , 'location']
numerical = ['number_of_courses_viewed', 'annual_income', 'interaction_count', 'lead_score']


#training


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model



def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


#validation

print(f'Doing validation C = {C}')

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
C= 1.0
scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.converted.values
    y_val = df_val.converted.values

    dv, model = train(df_train, y_train, C=1.0)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    print(f'auc on fold is {fold} on {auc}')
    fold = fold + 1

print('Validation result')
print('C=%4s, %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

#training the final model

print('training the final model')
dv, model = train(df_full_train,df_full_train.converted.values , C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.converted.values
auc = roc_auc_score(y_test, y_pred)
print(f'auc={auc}')

#saving the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv,model), f_out)

print(f'the model is saved to {output_file}')
