import logging


class ModelBuilder:

    def __init__(self, model_path):
        self.model_path = model_path

    def build_model(self):
        loan_history = []
