import logging


class Bidder:

    def __init__(self, min_investment, max_investment, api, loan_classifier):
        self.min_investment = min_investment
        self.max_investment = max_investment
        self.api = api
        self.loan_classifier = loan_classifier

    def _construct_bids(self, available_balance, attractive_auctions):
        logging.info("Available balance before bidding: {} EUR".format(available_balance))
        bids = []
        for auction in attractive_auctions:
            if available_balance < self.min_investment:
                break
            else:
                bid_amount = min(self.max_investment, available_balance)
                bids.append({
                    "AuctionId": auction['AuctionId'],
                    "MinAmount": self.min_investment,
                    "Amount": bid_amount
                })
                available_balance -= bid_amount
                logging.info("Bidding {} EUR with an expected return of {}%".format(bid_amount, auction['PredictedReturn'] * 100))
        logging.info("Estimated balance after bidding: {} EUR".format(available_balance))
        return bids

    def bid(self):
        logging.info("Bidding with min {} EUR and max {} EUR bids".format(self.min_investment, self.max_investment))
        available_balance = self.api.get_account_balance()['TotalAvailable']
        if available_balance < self.min_investment:
            logging.info("Insufficient funds for bidding: {} EUR".format(available_balance))
            return

        attractive_auctions = self.loan_classifier.find_attractive_auctions()
        bids = self._construct_bids(available_balance, attractive_auctions)
        if bids:
            self.api.post_bids(bids)
        logging.info("Bidding completed successfully")
