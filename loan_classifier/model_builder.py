import logging
from os import system
from datetime import datetime

import pandas
import numpy
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve


class ModelBuilder:
    LOAN_HISTORY_URL = "https://www.bondora.com/marketing/media/LoanData.zip"
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

    def __init__(self, model_path):
        self.model_path = model_path

    @staticmethod
    def _fetch_data():
        logging.info("Downloading loan history from {}".format(ModelBuilder.LOAN_HISTORY_URL))
        system("wget -O LoanData.zip {}".format(ModelBuilder.LOAN_HISTORY_URL))
        logging.info("Opening the downloaded loan history in LoanData.zip")
        system("unzip -o LoanData.zip && rm LoanData.zip")
        return pandas.read_csv("LoanData.csv")

    @staticmethod
    def _clean_data(data):
        logging.info("Removing unfinished loans from the loan history")
        data.MaturityDate_Last = data.MaturityDate_Last.astype('datetime64')
        data = data[data.MaturityDate_Last < datetime.today()]
        logging.info("Removing variables that are not used as predictors")
        input = data[ModelBuilder.PREDICTOR_VARIABLES]
        input[ModelBuilder.NUMERIC_VARIABLES] = input[ModelBuilder.NUMERIC_VARIABLES].astype('float64')
        input[ModelBuilder.CATEGORIC_VARIABLES] = input[ModelBuilder.CATEGORIC_VARIABLES].apply(lambda var: var.astype('category'))
        target = pandas.isnull(data.DefaultDate)
        logging.info("Cleaned loan history data has {} loans and {} predictors".format(len(input), len(ModelBuilder.PREDICTOR_VARIABLES)))
        return numpy.array(pandas.get_dummies(input)), numpy.array(target)

    @staticmethod
    def _log_results(model, test_data):
        predictions = model.predict(test_data)

    def build_model(self):
        logging.info("Starting to build the model")
        data = self._fetch_data()
        input, target = ModelBuilder._clean_data(data)
        train_input, test_input, train_target, test_target = train_test_split(input, target, test_size=0.1)
        dtrain = xgboost.DMatrix(train_input, train_target)
        dtest = xgboost.DMatrix(test_input, test_target)
        #Parameters obtained via the script parameter_tuning.py - too computationally expensive to tune every time
        params = {'objective': 'binary:logistic',
                  'eval_metric': 'auc',
                  'scale_pos_weight': (len(train_target) - sum(train_target))/sum(train_target),
                  'eta': 0.2,
                  'max_depth': 7,
                  'min_child_weight': 1}
        logging.info("Training XGBooster model with parameters: {}".format(params))
        model = xgboost.train(params, dtrain, num_boost_round=154)
        ModelBuilder._log_results(model, dtest)
        logging.info("Saving model in: {}".format(self.model_path))
        model.save_model(self.model_path)
