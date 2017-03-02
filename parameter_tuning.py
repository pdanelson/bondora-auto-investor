from datetime import datetime

import pandas
import numpy
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from loan_classifier.data_transformer import DataTransformer

# Preprocess data
data = pandas.read_csv("LoanData.csv")
data.LoanDate = data.LoanDate.astype("datetime64")
data = data[(data.LoanDate + numpy.timedelta64(2, 'Y') < datetime.today()) & (data.Country == "EE")]
input = DataTransformer().transform(data)
target = pandas.isnull(data.DefaultDate)

# Grid search to tune the parameters
scale_pos_weight = (len(target) - sum(target))/sum(target)  # nr of negative cases divided by nr of positive cases
model = xgboost.XGBClassifier(scale_pos_weight=scale_pos_weight)
cv_params = {"max_depth": [6, 7, 8, 9],
             "n_estimators": [200, 400, 600, 800],
             "min_child_weight": [1, 1.5],
             "gamma": [0, 0.01, 0.1],
             "learning_rate": [0.05, 0.1, 0.2]}
grid_search = GridSearchCV(model, cv_params, scoring="roc_auc", n_jobs=-1, cv=5, verbose=1)
grid_result = grid_search.fit(input, target)
joblib.dump(grid_result, "grid_result.pkl")

# Cross-validation with early stopping and best parameters from grid search to tune the number of boosting rounds
dtrain = xgboost.DMatrix(input, target)
params = {"objective": "binary:logistic",
          "scale_pos_weight": scale_pos_weight,
          "eta": 0.2,
          "max_depth": 8,
          "min_child_weight": 1.5,
          "gamma": 0}
cv_xgb = xgboost.cv(params, dtrain, num_boost_round=2000, nfold=5, metrics=["auc"], early_stopping_rounds=200)
