## What is it

A hobby project that aims to create a loan risk estimation model via machine learning and to use
this model to make automated investment decisions in the Bondora peer-to-peer lending portal.
Needless to say that I am not a financial advisor and you should not test it out with money you are not prepared to lose.
## How to use it
If you already have Python 3 installed, start by cloning the repo and installing the dependencies.
```
python3 -m pip install -r requirements.txt
```
Then configure the options in bondora.ini (explanations for the variables are commented in the file) and use Cron to
schedule the job of building the loan classification model from historical data (for example 3 AM on every first day
of the month, to update the model with new data).
```
0 3 1 * * python3 /path/to/bondora-auto-investor/main.py --build-model
```
Finally, set up the Cron job to execute the task of using the built loan classification model
(so make sure the previous job has been run at least once) to automatically invest into
loans suggested by the model via the Bondora API (for example, once every minute to constantly poll for new loan auctions).
```
* * * * * python3 /path/to/bondora-auto-investor/main.py --invest
