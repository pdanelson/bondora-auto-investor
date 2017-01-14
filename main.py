import logging
from sys import argv
from configparser import ConfigParser

from api.api import API
from bidder.bidder import Bidder
from loan_classifier.loan_classifier import LoanClassifier
from loan_classifier.model_builder import ModelBuilder


def invest(config):
    api = API(config["AccessToken"])
    loan_classifier = LoanClassifier(config["ProfitThreshold"], config["ModelPath"], api)
    bidder = Bidder(config["MinInvestment"], config["MaxInvestment"], api, loan_classifier)
    bidder.bid()


def build_model(config):
    model_builder = ModelBuilder(config["ModelPath"])
    model_builder.build_model()


def main(args):
    logging.basicConfig(filename="auto-investor.log",
                        format="%(asctime)s %(levelname)s: %(message)s",
                        level=logging.INFO)
    config = ConfigParser()
    config.read("bondora.ini")

    if "--invest" in args:
        logging.info("Investment mode chosen")
        invest(config["PRODUCTION"])
    elif "--build-model" in args:
        logging.info("Model building mode chosen")
        build_model(config["PRODUCTION"])
    else:
        logging.error("Invalid mode - specify either --invest or --build-model")
    return

if __name__ == "__main__":
    main(argv[1:])
