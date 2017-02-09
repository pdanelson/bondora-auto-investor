import logging

import xgboost
import pandas

from .data_transformer import DataTransformer


class LoanClassifier:

    def __init__(self, confidence_threshold, min_interest, model_path, api):
        self.confidence_threshold = confidence_threshold
        self.min_interest = min_interest
        self.model = xgboost.Booster(model_file=model_path)
        self.api = api

    def _assign_confidence_level(self, auctions):
        input = DataTransformer().transform(pandas.DataFrame.from_records(auctions))
        predictions = self.model.predict(xgboost.DMatrix(input))
        return [dict(auction, Confidence=prediction) for auction, prediction in zip(auctions, predictions)]

    def _has_any_prior_bids(self, auction, bids):
        return any(bid["AuctionId"] == auction["AuctionId"] for bid in bids)

    def _is_attractive_auction(self, auction):
        return auction["Confidence"] > self.confidence_threshold and auction["Interest"] > self.min_interest

    def find_attractive_auctions(self):
        bids = self.api.get_bids()
        auctions = self.api.get_auctions()
        available_auctions = [auction for auction in auctions if not self._has_any_prior_bids(auction, bids)]
        logging.info("Nr of all available auctions I have not made bids on: {}".format(len(available_auctions)))
        if not available_auctions:
            return []
        evaluated_auctions = self._assign_confidence_level(available_auctions)
        attractive_auctions = [auction for auction in evaluated_auctions if self._is_attractive_auction(auction)]
        logging.info("Nr of available auctions exceeding the confidence and interest rate thresholds: {}"
                     .format(len(attractive_auctions)))
        return sorted(attractive_auctions, key=lambda auction: auction["Interest"], reverse=True)
