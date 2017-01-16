from datetime import datetime

import pandas
import numpy
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


CATEGORIC_VARIABLES = ['Country', 'CreditScoreEeMini', 'CreditScoreEsEquifaxRisk', 'CreditScoreEsMicroL',
                       'CreditScoreFiAsiakasTietoRiskGrade', 'Education', 'EmploymentDurationCurrentEmployer',
                       'EmploymentStatus', 'Gender', 'HomeOwnershipType', 'LanguageCode',
                       'MaritalStatus', 'MonthlyPaymentDay', 'NewCreditCustomer', 'OccupationArea', 'Rating',
                       'UseOfLoan', 'VerificationType', 'NrOfDependants', 'WorkExperience']
NUMERIC_VARIABLES = ['Age', 'AppliedAmount', 'DebtToIncome', 'ExpectedLoss', 'LiabilitiesTotal', 'FreeCash',
                     'IncomeFromChildSupport', 'IncomeFromFamilyAllowance', 'IncomeFromLeavePay',
                     'IncomeFromPension', 'IncomeFromPrincipalEmployer', 'IncomeFromSocialWelfare', 'IncomeOther',
                     'IncomeTotal', 'Interest', 'LoanDuration', 'LossGivenDefault', 'MonthlyPayment',
                     'ProbabilityOfDefault']
PREDICTOR_VARIABLES = CATEGORIC_VARIABLES + NUMERIC_VARIABLES

#Preprocess data
data = pandas.read_csv("LoanData.csv")
data.MaturityDate_Last = data.MaturityDate_Last.astype('datetime64')
data = data[data.MaturityDate_Last < datetime.today()]
input = data[PREDICTOR_VARIABLES]
input[NUMERIC_VARIABLES] = input[NUMERIC_VARIABLES].astype('float64')
input[CATEGORIC_VARIABLES] = input[CATEGORIC_VARIABLES].apply(lambda var: var.astype('category'))
target = pandas.isnull(data.DefaultDate)
input, target = numpy.array(pandas.get_dummies(input)), numpy.array(target)
train_input, test_input, train_target, test_target = train_test_split(input, target, test_size=0.2)

#Grid search to tune the parameters
scale_pos_weight = (len(train_target) - sum(train_target))/sum(train_target) #nr of negative cases divided by nr of positive cases
model = xgboost.XGBClassifier(scale_pos_weight=scale_pos_weight)
cv_params = {'max_depth': [4, 5, 6, 7],
             'n_estimators': [100, 200, 300, 400],
             'min_child_weight': [1, 2, 3, 4],
             'learning_rate': [0.05, 0.1, 0.2]}
grid_search = GridSearchCV(model, cv_params, scoring="roc_auc", n_jobs=-1, cv=5, verbose=1)
grid_result = grid_search.fit(train_input, train_target)
joblib.dump(grid_result, "grid_result.pkl")

#Crossvalidation with early stopping and best parameters from grid search to tune the number of boosting rounds
dtrain = xgboost.DMatrix(train_input, train_target)
dtest = xgboost.DMatrix(test_input, test_target)
params = {'objective': 'binary:logistic',
          'scale_pos_weight': scale_pos_weight,
          'eta': 0.2,
          'max_depth': 7,
          'min_child_weight': 1}
cv_xgb = xgboost.cv(params, dtrain, num_boost_round = 2000, nfold = 5, metrics = ['auc'], early_stopping_rounds = 100)