# Monetary-Fraud-Detection-System-

Dataset - https://www.kaggle.com/datasets/ealaxi/paysim1 

Attribute Info - 
TRANSFER - 0
CASH_OUT - 1

Input Attributes -
step  
type        
amount  
oldbalanceOrg  
newbalanceOrig
oldbalanceDest  
newbalanceDest
isFlaggedFraud 

Derived Attributes
errorBalanceOrig  - newBalanceOrig - oldBalanceOrig
errorBalanceDest  - newBalanceDest - oldBalanceDest

To Predict - isFraud