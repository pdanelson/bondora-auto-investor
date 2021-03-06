import logging
import numpy
import pandas
import xgboost
from datetime import datetime
from operator import itemgetter
from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen
from sklearn.model_selection import train_test_split

from .data_transformer import DataTransformer


class ModelBuilder:
    LOAN_HISTORY_URL = "https://www.bondora.com/marketing/media/LoanData.zip"

    def __init__(self, model_path):
        self.model_path = model_path

    def _fetch_data(self):
        logging.info("Downloading loan history from {}".format(self.LOAN_HISTORY_URL))
        zip_file = ZipFile(BytesIO(urlopen(self.LOAN_HISTORY_URL).read()))
        csv_file = zip_file.open("LoanData.csv")
        return pandas.read_csv(csv_file)

    @staticmethod
    def _prepare_data(data):
        data.LoanDate = data.LoanDate.astype("datetime64")
        data = data[(data.LoanDate + numpy.timedelta64(2, 'Y') < datetime.today()) & (data.Country == "EE")]
        logging.info("Cleaned loan history data has {} loans".format(len(data)))
        return DataTransformer.transform(data), pandas.isnull(data.DefaultDate)

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
                logging.info("MODEL VALIDATION RESULTS for CONFIDENCE THRESHOLD {:.2} and MINIMUM INTEREST {}%:"
                             .format(confidence_threshold, min_interest))
                confident_predictions = numpy.where(predictions > confidence_threshold)[0]
                chosen_loans = test_input.loc[confident_predictions, :]
                chosen_loans = chosen_loans[chosen_loans.Interest > min_interest]
                chosen_loans["Repaid"] = test_target[confident_predictions]
                logging.info("Invested into {} loans out of a possible {} ({:.2%})"
                             .format(len(chosen_loans), len(test_input), len(chosen_loans) / len(test_input)))
                if chosen_loans.empty:
                    continue
                # Conservative estimate of losing all principal on a default and paying 20% income tax on interest
                loan_outcomes = chosen_loans.Interest/100 * 0.8 * chosen_loans.Repaid - (1 - chosen_loans.Repaid)
                best_outcomes = sorted(loan_outcomes, reverse=True)[0:50]
                worst_outcomes = sorted(loan_outcomes)[0:50]
                logging.info("In hindsight, {} ({:.2%}) defaulted, resulting in:"
                             .format(len(chosen_loans) - sum(chosen_loans.Repaid),
                                     (len(chosen_loans) - sum(chosen_loans.Repaid)) / len(chosen_loans)))
                logging.info("Yearly after tax return of {:.2%} overall".format(numpy.mean(loan_outcomes)))
                logging.info("Yearly after tax return of {:.2%} for best 50 loans".format(numpy.mean(best_outcomes)))
                logging.info("Yearly after tax return of {:.2%} for worst 50 loans".format(numpy.mean(worst_outcomes)))

    def build_model(self):
        logging.info("Starting to build the model")
        data = self._fetch_data()
        input, target = ModelBuilder._prepare_data(data)
        train_input, test_input, train_target, test_target = train_test_split(input, target, test_size=0.2)
        # Parameters obtained via the script parameter_tuning.py - too computationally expensive to tune every time
        params = {"objective": "binary:logistic",
                  "eval_metric": "auc",
                  "scale_pos_weight": (len(train_target) - sum(train_target))/sum(train_target),
                  "eta": 0.2,
                  "max_depth": 8,
                  "min_child_weight": 1.5}
        logging.info("Training XGBooster model with parameters:\n{}".format(params))
        model = xgboost.train(params, xgboost.DMatrix(train_input, train_target), num_boost_round=212)
        ModelBuilder._log_model_statistics(model, test_input, test_target)
        model.save_model(self.model_path)
        logging.info("Saved model as: {}".format(self.model_path))
