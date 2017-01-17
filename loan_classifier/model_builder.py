import logging
from operator import itemgetter
from os import system
from datetime import datetime

import numpy
import pandas
import xgboost
from sklearn.model_selection import train_test_split


class ModelBuilder:
    LOAN_HISTORY_URL = "https://www.bondora.com/marketing/media/LoanData.zip"
    CATEGORIC_VARIABLES = ["Country", "CreditScoreEeMini", "CreditScoreEsEquifaxRisk", "CreditScoreEsMicroL",
                           "CreditScoreFiAsiakasTietoRiskGrade", "Education", "EmploymentDurationCurrentEmployer",
                           "EmploymentStatus", "Gender", "HomeOwnershipType", "LanguageCode",
                           "MaritalStatus", "MonthlyPaymentDay", "NewCreditCustomer", "OccupationArea", "Rating",
                           "UseOfLoan", "VerificationType", "NrOfDependants", "WorkExperience"]
    NUMERIC_VARIABLES = ["Age", "AppliedAmount", "DebtToIncome", "ExpectedLoss", "LiabilitiesTotal", "FreeCash",
                         "IncomeFromChildSupport", "IncomeFromFamilyAllowance", "IncomeFromLeavePay",
                         "IncomeFromPension", "IncomeFromPrincipalEmployer", "IncomeFromSocialWelfare", "IncomeOther",
                         "IncomeTotal", "Interest", "LoanDuration", "LossGivenDefault", "MonthlyPayment",
                         "ProbabilityOfDefault"]
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
        data.MaturityDate_Last = data.MaturityDate_Last.astype("datetime64")
        data = data[data.MaturityDate_Last < datetime.today()]
        logging.info("Removing variables that are not used as predictors")
        input = data[ModelBuilder.PREDICTOR_VARIABLES]
        input[ModelBuilder.NUMERIC_VARIABLES] = input[ModelBuilder.NUMERIC_VARIABLES].astype("float64")
        input[ModelBuilder.CATEGORIC_VARIABLES] = input[ModelBuilder.CATEGORIC_VARIABLES].apply(lambda var: var.astype("category"))
        target = pandas.isnull(data.DefaultDate)
        logging.info("Cleaned loan history data has {} loans and {} predictors".format(len(input), len(ModelBuilder.PREDICTOR_VARIABLES)))
        return pandas.get_dummies(input), target

    @staticmethod
    def _log_model_statistics(model, test_input, test_target):
        feature_importance = sorted(model.get_fscore().items(), key=itemgetter(1), reverse=True)
        importance_sum = sum(importance for feature, importance in feature_importance)
        rel_feature_importance = [(feature, importance / importance_sum) for feature, importance in feature_importance]
        logging.info("The 10 most important data features in the model are:\n{}".format(rel_feature_importance[0:10]))
        predictions = model.predict(xgboost.DMatrix(test_input, test_target))
        test_input.reset_index(inplace=True, drop=True)
        test_target.reset_index(inplace=True, drop=True)
        for confidence_threshold in numpy.arange(0.9, 1, 0.01):
            for min_interest in range(15, 31, 5):
                logging.info("Model validation results for confidence threshold {:.2} and minimum interest {}%"
                             .format(confidence_threshold, min_interest))
                confident_predictions = numpy.where(predictions > confidence_threshold)[0]
                chosen_loans = test_input.loc[confident_predictions, :]
                chosen_loans = chosen_loans[chosen_loans.Interest > min_interest]
                chosen_loans["Repaid"] = test_target[confident_predictions]
                logging.info("Invested into {} loans out of a possible {} ({:.2%})"
                             .format(len(chosen_loans), len(test_input), len(chosen_loans) / len(test_input)))
                # Conservative estimate of losing all principal on a default and paying 20% income tax on interest
                loan_outcomes = chosen_loans.Interest/100 * 0.8 * chosen_loans.Repaid - (1 - chosen_loans.Repaid)
                logging.info("In hindsight, {} ({:.2%}) defaulted, resulting in a yearly after tax return of {:.2%}"
                             .format(len(chosen_loans) - sum(chosen_loans.Repaid),
                                     (len(chosen_loans) - sum(chosen_loans.Repaid)) / len(chosen_loans),
                                     numpy.mean(loan_outcomes)))

    def build_model(self):
        logging.info("Starting to build the model")
        data = ModelBuilder._fetch_data()
        input, target = ModelBuilder._clean_data(data)
        train_input, test_input, train_target, test_target = train_test_split(input, target, test_size=0.2)
        # Parameters obtained via the script parameter_tuning.py - too computationally expensive to tune every time
        params = {"objective": "binary:logistic",
                  "eval_metric": "auc",
                  "scale_pos_weight": (len(train_target) - sum(train_target))/sum(train_target),
                  "eta": 0.2,
                  "max_depth": 7,
                  "min_child_weight": 1}
        logging.info("Training XGBooster model with parameters:\n{}".format(params))
        model = xgboost.train(params, xgboost.DMatrix(train_input, train_target), num_boost_round=154)
        ModelBuilder._log_model_statistics(model, test_input, test_target)
        model.save_model(self.model_path)
        logging.info("Saved model as: {}".format(self.model_path))
