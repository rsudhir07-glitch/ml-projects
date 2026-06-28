import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report , confusion_matrix
df = pd.read_csv('salary_prediction/data/employeedata.csv')

# handling data 

df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
df.replace('?', np.nan , inplace = True)

for col in ['workclass', 'occupation']:
    df[col] = df[col].fillna(df[col].mode()[0])

# dropping unwanted columns 

df = df.drop(columns = ['education'])

# encoding dataset

df['sex'] = df['sex'].map({
    'Male' : 1,
    'Female' : 0,
})
df['income'] = df['income'].map({
    '>50K' : 1 ,
    '<=50K' : 0
})

df = pd.get_dummies(
    df,
    columns = [
        'workclass',
        'marital.status',
        'relationship',
        'occupation',
        'race',
        'native.country'
    ]
)

'''

after using dummies on columns, they no longer exist like before, thats why while difining features we will jus drop theh target column , so that the remaning ( dummies ) get selected as features

'''

# features and target

X  = df.drop(columns = ['income'])
Y = df['income']

# training the model

# here we will use classification model which predicts whether the income is more than 50k or <=50k

X_train, X_test , Y_train , Y_test = train_test_split(
    X ,Y ,
    test_size = 0.3,
    random_state = 42
)

model = DecisionTreeClassifier()

# decision trees do not need scaling 

model.fit(X_train , Y_train )
y_pred = model.predict(X_test)

# model evaluation 

print(classification_report(Y_test , y_pred)  )    
cm = confusion_matrix(Y_test , y_pred)
print(cm)