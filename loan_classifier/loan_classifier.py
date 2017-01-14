import logging

import pickle


class LoanClassifier:

    def __init__(self, profit_threshold, model_path, api):
        self.profit_threshold = profit_threshold
        self.model_path = model_path
        self.api = api

    def find_attractive_auctions(self):
        auctions = self.api.get_auctions()
        model = pickle.loads(self.model_path)

