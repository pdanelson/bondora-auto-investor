import requests


class API:
    BASE_URL = "https://api.bondora.com/api/v1"

    def __init__(self, token):
        self.headers = {"Authorization": "Bearer {}".format(token), "Content-Type": "application/json"}

    def _get(self, url):
        response = requests.get("{}/{}".format(API.BASE_URL, url), headers=self.headers)
        if response.ok:
            return response.json()["Payload"]
        raise Exception(response.json()["Error"])

    def _post(self, url, payload):
        response = requests.post("{}/{}".format(API.BASE_URL, url), json=payload, headers=self.headers)
        if response.ok:
            return response.json()["Payload"]
        raise Exception(response.json()["Error"])

    def get_account_balance(self):
        return self._get("account/balance")

    def post_bids(self, bids):
        return self._post("bid", {"Bids": bids})

    def get_auctions(self):
        return self._get("auctions")

    def get_bids(self):
        return self._get("bids")
