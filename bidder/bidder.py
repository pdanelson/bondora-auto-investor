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
                # Rounded down to nearest multiple of 5
                amount = min(self.max_investment, available_balance) - (min(self.max_investment, available_balance) % 5)
                bids.append({
                    "AuctionId": auction["AuctionId"],
                    "MinAmount": self.min_investment,
                    "Amount": amount
                })
                available_balance -= amount
                logging.info("Bidding {} EUR into a {} loan with {} rating and {}% interest rate"
                             .format(amount, auction["Country"], auction["Rating"], auction["Interest"]))
        logging.info("Estimated balance after bidding: {} EUR".format(available_balance))
        return bids

    def bid(self):
        logging.info("Bidding with min {} EUR and max {} EUR bids".format(self.min_investment, self.max_investment))
        available_balance = self.api.get_account_balance()["TotalAvailable"]
        if available_balance < self.min_investment:
            logging.info("Insufficient funds for bidding: {} EUR".format(available_balance))
            return
        attractive_auctions = self.loan_classifier.find_attractive_auctions()
        bids = self._construct_bids(available_balance, attractive_auctions)
        if bids:
            self.api.post_bids(bids)
        logging.info("Bidding completed successfully")
