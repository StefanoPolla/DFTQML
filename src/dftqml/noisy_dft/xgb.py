from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.utils import shuffle
from sklearn.model_selection import cross_validate
import os
from scipy.stats import uniform, randint

from os import path 
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
import joblib 
from xgboost.sklearn import XGBRegressor

def model_path(L, N, U, source, ndata, split):
    return path.join("models/xgb/", f"L{L}-N{N}-U{U}", source, f"ndata{ndata}", f"split{split}")

def initialize_xgb(**parameters):
    model = XGBRegressor(**parameters)
    return model


def fit_xgb(model, x_train, y_train, optimize_hyperparameters, set_shuffle, verbose):
    if set_shuffle:
        x_train, y_train = shuffle(x_train, y_train)
        
    if optimize_hyperparameters:
        
        parameters = {'n_estimators':randint(100,800),
                'learning_rate':uniform(0.0,0.4), #base:0.3
                'min_split_loss':randint(0,3), #base:0
                'max_depth':randint(3,10), #base:6
                'subsample':uniform(0.5,0.5), #base:1
                'lambda':uniform(0,2), #base:1
                'min_child_weight':randint(1,6) #base:1
                } 
        
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        grid_obj = HalvingRandomSearchCV(model, parameters, scoring=scorer,n_jobs=-1, cv=5, verbose=verbose,factor=3)
        
        # Fit the grid search object to the training data and find the optimal parameters using fit()
        grid_fit = grid_obj.fit(x_train, y_train)

        # Get the best model estimator
        best_model = grid_fit.best_estimator_
        
        model_obj = cross_validate(
            best_model,
            x_train,
            y_train,
            cv=5,
            scoring="neg_mean_squared_error",
            return_estimator=True,
            verbose=verbose,
        )

    else:
        model_obj = cross_validate(
            model,
            x_train,
            y_train,
            cv=5,
            scoring="neg_mean_squared_error",
            return_estimator=True,
            verbose=verbose,
        )

    return model_obj

def save_xgb(cv_model, path):
    for k, k_model in enumerate(cv_model["estimator"]):
        output_path = os.path.join(path, f"split{k}" + "_model.pkl")
        # k_model.save_model(output_path)
        
        joblib.dump(k_model, output_path) 

    # with open(os.path.join(path, "xgb_hyperparameters.json"), "w") as f:
    #     json.dump(cv_model["estimator"][0].get_params(), f)
        
def load_xgb(model_path):
    # hyp_path = re.sub(r'/split\d+', '', model_path)
    # with open(hyp_path+'/xgb_hyperparameters.json', 'r') as f:
    #     params = json.load(f)

    # # Initialize a new model with hyperparameters
    # model = initialize_xgb(**params)    
    # model.load_model(model_path+'_model.json')
    # print("Trying to load",model_path)
    # with open(model_path+'_model.json', 'rb') as file:
    #     s = file.read()
    
    return joblib.load(model_path+'_model.pkl')