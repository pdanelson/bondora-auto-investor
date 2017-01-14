import requests
import json


class API:
    BASE_URL = "https://api.bondora.com/api/v1"

    def __init__(self, token):
        self.headers = {"Authorization": "Bearer {}".format(token), "Content-Type": "application/json"}

    def get(self, url):
        response = requests.get("{}/{}".format(API.BASE_URL, url), headers=self.headers)
        if response.ok:
            return response.json()['Payload']
        raise Exception(response.json()['Error'])

    def post(self, url, payload):
        response = requests.post("{}/{}".format(API.BASE_URL, url), json.dumps(payload), headers=self.headers)
        if response.ok:
            return response.json()['Payload']
        raise Exception(response.json()['Error'])

    def get_account_balance(self):
        return self.get("account/balance")

    def get_bids(self):
        return self.get("bids")

    def post_bids(self, bids):
        return self.post("bid", bids)

    def get_auctions(self):
        return self.get("auctions")

    def get_loans_history(self):
        return self.get("loandataset")
