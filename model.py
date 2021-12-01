import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTEENN
import pickle

popchurn_df = pd.read_csv('C:/Python/Ageing dump.csv')
popchurn = popchurn_df.copy(deep = True)


popchurn.drop(['PYMNT REM', 'PRG NAME', 'RATE PLAN', 'ZIP'], inplace = True, axis = 1)
#popchurn['OLDEST_BUCKET_NEW'] = pd.to_numeric(popchurn['OLDEST_BUCKET_NEW'], errors = 'coerce')

print(popchurn['CHURN'].value_counts())
print(100*popchurn['CHURN'].value_counts()/len(popchurn['CHURN']))

list_obj_col = list(popchurn.select_dtypes(include = 'object').columns)
list_num_col = list(popchurn.select_dtypes(exclude = 'object').columns)

def fillna(df):
    for i in list_obj_col:
        df[i].fillna(value= df[i].mode()[0], inplace = True)
    for j in list_num_col:
        df[j].fillna(value = df[j].mean(), inplace = True)
fillna(popchurn)

popchurn.isnull().sum()

popchurn_num = popchurn.select_dtypes(include = [np.number])
popchurn_cat = popchurn.select_dtypes(exclude = [np.number])

labelencode = LabelEncoder()
mapping = {}
for i in popchurn_cat:
    popchurn[i] = labelencode.fit_transform(popchurn[i])
    le_name_mapping = dict(zip(labelencode.classes_, labelencode.transform(labelencode.classes_)))
    mapping[i]=le_name_mapping
print(mapping)


X = popchurn.drop(['CHURN'], axis = 1)
y = popchurn['CHURN']
print(X.shape)

#SMOTEENN
sm = SMOTEENN()
X_resample, y_resample = sm.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resample, y_resample, test_size = .3, random_state = 100)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print(confusion_matrix(y_test, pred))

#saving the model
pickle.dump(model, open('model_popchurn.pkl', 'wb'))

#check the prediction on a test data
print(model.predict([[0, 5, 0, 508, 8, 2500, 27, 29, 1, 0, 0, 0, 0, 3, 1, 1,0,220, 399]]))
