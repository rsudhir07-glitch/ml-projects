from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def trained_model(data):
    X = data.drop(columns = ['Response'])
    y = data['Response']
    X_train , X_test , y_train , y_test = train_test_split(
       X , y,
       test_size= 0.3,
       random_state = 42,
       stratify = y

    )
    
    model = XGBClassifier(random_state = 42)


    # now we will define hyperparameters

    params = {
        'n_estimators' : [100 , 200 , 50],
        'max_depth' : [3 , 5],
        'learning_rate' : [0.1 , 0.05],
        'subsample' : [0.5 , 0.8],
        'colsample_bytree' : [0.5 , 0.8]
  
    }

    cv = StratifiedKFold(
        n_splits= 5,
        shuffle=True,
        random_state = 42

    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv = cv,
        n_jobs =-1,
        scoring = 'f1'
    )

    grid.fit(X_train , y_train)
    print("Best Parameters:", grid.best_params_)
    print("Best CV Score:", grid.best_score_)

    best_model = grid.best_estimator_

    
    return best_model , X_test , y_test

