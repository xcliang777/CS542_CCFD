# CS542_CCFD

# Credit Card Fraud Detection
It is important to recognize fraudulent credit card transactions so that customers are not charged for transaction that they did not make.

In this problem, we use three different ways: Logistic regression, Random forest and Decision tree.

# Data
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset have 492 frauds out of 284,807 transactions in two days. 

The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. If we just use this data to do the training and testing, we will still get a result that most are labeled as no fraud. As a result, the accuracy will be high. But we can not tell the model is doing good. So we have to preprocess the data and make it better for the model to train. 

## How to run the code
Please download the dataset first following the link below and put it in the same directory of our code.
https://www.kaggle.com/mlg-ulb/creditcardfraud
```
python3 main.py
```
