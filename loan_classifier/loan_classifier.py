import logging

import xgboost
import pandas

from .model_builder import ModelBuilder


class LoanClassifier:

    def __init__(self, profit_threshold, model_path, api):
        self.profit_threshold = profit_threshold
        self.model = xgboost.Booster(model_file=model_path)
        self.api = api

    def _assign_predicted_return(self, auctions):
        input = pandas.DataFrame.from_records(auctions)[ModelBuilder.PREDICTOR_VARIABLES]
        predictions = self.model.predict(xgboost.DMatrix(input))
        # Conservative estimate of losing 100% principal when a loan defaults and 20% tax on profits when it doesn't
        auctions['PredictedReturn'] = predictions * auctions['Interest'] * 0.8 - (1 - predictions)
        return auctions

    def find_attractive_auctions(self):
        available_auctions = [auction for auction in self.api.get_auctions() if auction['UserBids'] < 1]
        logging.info("Nr of all available auctions I have not made bids on: {}".format(len(available_auctions)))
        evaluated_auctions = self._assign_predicted_return(available_auctions)
        attractive_auctions = [auction for auction in evaluated_auctions if auction['PredictedReturn'] > self.profit_threshold]
        logging.info("Nr of available auctions matching the expected profit threshold: {}".format(len(attractive_auctions)))
        return sorted(attractive_auctions, key=lambda auction: auction['PredictedReturn'], reversed=True)
