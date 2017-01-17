import logging

import xgboost
import pandas

from .model_builder import ModelBuilder


class LoanClassifier:

    def __init__(self, confidence_threshold, min_interest, model_path, api):
        self.confidence_threshold = confidence_threshold
        self.min_interest = min_interest
        self.model = xgboost.Booster(model_file=model_path)
        self.api = api

    def _assign_confidence_level(self, auctions):
        input = pandas.DataFrame.from_records(auctions)[ModelBuilder.PREDICTOR_VARIABLES]
        predictions = self.model.predict(xgboost.DMatrix(input))
        auctions["Confidence"] = predictions
        return auctions

    def _is_attractive_auction(self, auction):
        return auction["Confidence"] > self.confidence_threshold and auction["Interest"] > self.min_interest

    def find_attractive_auctions(self):
        available_auctions = [auction for auction in self.api.get_auctions() if auction["UserBids"] < 1]
        logging.info("Nr of all available auctions I have not made bids on: {}".format(len(available_auctions)))
        evaluated_auctions = self._assign_confidence_level(available_auctions)
        attractive_auctions = [auction for auction in evaluated_auctions if self._is_attractive_auction(auction)]
        logging.info("Nr of available auctions exceeding the confidence and interest rate thresholds: {}"
                     .format(len(attractive_auctions)))
        return sorted(attractive_auctions, key=lambda auction: auction["Interest"], reversed=True)
