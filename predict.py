import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler

bst = XGBClassifier()
x_test = [[1,1,9839.64,170136.0,160296.36,0.0,0.0,0]]

sc = StandardScaler()

# # x_train = sc.fit_transform(x_train)
# x_test = sc.fit_transform(x_test)
# print(x_test)

bst.load_model('1.model') 
print(bst.predict(x_test))