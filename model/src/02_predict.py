import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
# %matplotlib inline
mpl.rcParams["figure.figsize"] = (20,10)

df1 = pd.read_pickle("../data/01_data_processed.pkl")

df2 = df1.drop(['size','price_per_sqft'],axis='columns')
# Size is redundant as its the same as bhk
# price/sqft was only needed for oulier detection
df2


dummies = pd.get_dummies(df2.location) # one-hot encoding

df3 = pd.concat([df2,dummies.drop('other',axis='columns')], axis=1)
# I am dropping "other" column as we can use other columns to get it
# It is a standard practice while dealing with one-hot encoding

df3

df4 = df3.drop(["location"],axis="columns")

X = df4.drop(["price"], axis=1)
y = df4.price

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)

# Use K Fold cross validation to measure accuracy of our LinearRegression model

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# cv - Cross Validation
cross_val_score(LinearRegression(), X, y, cv=cv)



# Find best model using GridSearchCV


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

# def find_best_model_using_gridsearchcv(X,y):
#     algos = {
#         'linear_regression' : {
#             # 'model': LinearRegression(),
#             'model': Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('regressor', LinearRegression())
#             ]),
#             'params': {
#                 # 'normalize': [True, False] --- Deprecated
                
#                 # 'copy_X' : [True, False],
#                 # 'fit_intercept' : [True, False],
#                 # 'n_jobs' : [1,2,3],
#                 # 'positive' : [True, False]
#                 ('scaler', StandardScaler()),
#                 ('regressor', LinearRegression())
#             }
#         },
#         'lasso': {
#             'model': Lasso(),
#             'params': {
#                 'alpha': [1,2],
#                 'selection': ['random', 'cyclic']
#             }
#         },
#         'decision_tree': {
#             'model': DecisionTreeRegressor(),
#             'params': {
#                 'criterion' : ['mse','friedman_mse'],
#                 'splitter': ['best','random']
#             }
#         }
#     }
#     scores = []
#     cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
#     for algo_name, config in algos.items():
#         gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
#         gs.fit(X,y)
#         scores.append({
#             'model': algo_name,
#             'best_score': gs.best_score_,
#             'best_params': gs.best_params_
#         })

#     return pd.DataFrame(scores,columns=['model','best_score','best_params'])

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ]),
            'params': {
                'regressor__copy_X': [True, False],
                'regressor__fit_intercept': [True, False],
                'regressor__n_jobs': [1, 2, 3],
                'regressor__positive': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])


find_best_model_using_gridsearchcv(X,y)
# Based on above results we can say that LinearRegression gives the best score. Hence we will use that.



# Test the model for few properties

def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]



predict_price('1st Phase JP Nagar',1000, 2, 2)
predict_price('1st Phase JP Nagar',1000, 3, 3)
predict_price('Indira Nagar',1000, 2, 2)
predict_price('Indira Nagar',1000, 3, 3)



# Export the tested model to a pickle file
import pickle
with open('../final_model/banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)
    
    
# Export location and column information to a file that will be useful later on in our prediction application

import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("../final_model/columns.json","w") as f:
    f.write(json.dumps(columns))