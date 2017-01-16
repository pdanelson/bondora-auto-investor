import logging

import xgboost
import pandas

from .model_builder import ModelBuilder


class LoanClassifier:

    def __init__(self, confidence_threshold, model_path, api):
        self.confidence_threshold = confidence_threshold
        self.model = xgboost.Booster(model_file=model_path)
        self.api = api

    def _assign_confidence_level(self, auctions):
        input = pandas.DataFrame.from_records(auctions)[ModelBuilder.PREDICTOR_VARIABLES]
        predictions = self.model.predict(xgboost.DMatrix(input))
        auctions['Confidence'] = predictions
        # Amongst loans deemed sufficiently safe we want to achieve maximum income
        return sorted(auctions, key=lambda auction: auction['Interest'], reversed=True)

    def find_attractive_auctions(self):
        available_auctions = [auction for auction in self.api.get_auctions() if auction['UserBids'] < 1]
        logging.info("Nr of all available auctions I have not made bids on: {}".format(len(available_auctions)))
        evaluated_auctions = self._assign_confidence_level(available_auctions)
        attractive_auctions = [auction for auction in evaluated_auctions if auction['Confidence'] > self.confidence_threshold]
        logging.info("Nr of available auctions exceeding the confidence threshold: {}".format(len(attractive_auctions)))
        return attractive_auctions
