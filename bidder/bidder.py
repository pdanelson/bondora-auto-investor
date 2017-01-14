from api.api import API
import logging

class Bidder:

    def __init__(self, token, min_investment, max_investment, loan_classifier):
        self.api = API(token)
        self.min_investment = min_investment
        self.max_investment = max_investment
        self.loan_classifier = loan_classifier

    def _find_attractive_auctions(self):
        auctions = self.api.get_auctions()
        return self.loan_classifier.find_profitable_loans(auctions)

    def _construct_bids(self, available_balance, attractive_auctions):
        logging.info("Available balance before bidding: {} EUR".format(available_balance))
        bids = []
        for auction in attractive_auctions:
            if available_balance < self.min_investment:
                break
            elif auction['RemainingAmount'] < self.min_investment or auction['UserBids']:
                continue
            else:
                bid_amount = min(auction['RemainingAmount'], self.max_investment, available_balance)
                bids.append({
                    "AuctionId": auction['AuctionId'],
                    "MinAmount": self.min_investment,
                    "Amount": bid_amount
                })
                available_balance -= bid_amount
        logging.info("Available balance after bidding: {} EUR".format(available_balance))
        return bids

    def bid(self):
        logging.info("Bidding with min {} EUR and max {} EUR bids".format(self.min_investment, self.max_investment))
        available_balance = self.api.get_account_balance()['TotalAvailable']
        if available_balance < self.min_investment:
            logging.info("Insufficient funds for bidding: {} EUR".format(available_balance))
            return

        attractive_auctions = self._find_attractive_auctions()
        logging.info("Nr of attractive auctions found: {}".format(len(attractive_auctions)))
        bids = self._construct_bids(available_balance, attractive_auctions)
        logging.info("Nr of bids made: {}".format(len(bids)))
        if bids:
            self.api.post_bids(bids)
