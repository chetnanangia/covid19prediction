import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import joblib


def cough_converter(x):
    try:
        return Int32(x)
    except:
        return None

def convertor(x):
    try:
       retval = np.int32(x)
    except :
       retval = None
    return retval

def genderconvertor(g):
    if g == 'female':
        return np.int32(0)
    elif g == 'male':
        return np.int32(1)
    else:
        return None
   

def corona_result_convertor(r):
    if r == 'negative':
        return np.int32(0)
    elif r == 'positive':
        return np.int32(1)
    else:
        return None 

#def corona_result_convertor(r):
#    if r == 'negative':
#        return np.int32(0)
#    elif r == 'positive':
#        return np.int32(1)
#    else:
#        return None 

def age_convertor(c):
    if c == 'Yes':
        return np.int32(1)
    elif c == 'No':
        return np.int32(0)
    else:
        return None

def contact_convertor(c):
    if c == 'Other':
        return np.int32(0)
    elif c == 'Abroad':
        return np.int32(1)
    elif c == 'Contact with confirmed':
        return np.int32(2)

parse_dates = ['test_date']
converters = {'cough': convertor,   
              'fever': convertor, 'sore_throat': convertor,
              'shortness_of_breath': convertor, 'head_ache': convertor,'age_60_and_above': age_convertor, 
              'gender': genderconvertor,
              'corona_result' : corona_result_convertor,  
              'test_indication' : contact_convertor}

orig_df = pd.read_csv("corona_tested_individuals_ver_006.english.csv", 
                 parse_dates=parse_dates, converters=converters, low_memory=False)


print(orig_df.info())


# Cleanup the data 

df = orig_df.copy()
df.dropna(inplace=True)


Input = df.drop(['test_date', 'corona_result'], axis=1)
output = df['corona_result']

X_train, X_test, y_train, y_test = train_test_split(Input, output, test_size=0.2)


model = DecisionTreeClassifier()
model.fit(X_train.values, y_train)





predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions) 
print("score = %d", score) 
joblib.dump(model, "saved-model.joblib") 


model.predict([[1.,1.,1.,1.,0.,1.,1.,0]])

for col in ['cough', 'fever', 'sore_throat', 'shortness_of_breath',
       'head_ache', 'corona_result', 'age_60_and_above', 'gender',
       'test_indication']:
    print(col, df[col].unique())

test_input = {'cough' : 1, 'fever' :0, 'sore_throat' :1, 'shortness_of_breath':1,
       'head_ache' :0,  'age_60_and_above' :1, 'gender' :1,
       'test_indication':1}


t = tree.export_graphviz(model, out_file='covid.dot',
                          feature_names=['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache', 
                                         'age_60_and_above', 'gender', 'test_indication'], 
                          class_names=[str(x) for x in list(sorted(y_train.unique()))], label='all', rounded=True, filled=True)


reloaded = joblib.load('saved-model.joblib')

reloaded.predict([[1.,1.,1.,1.,0.,1.,1.,0]])


