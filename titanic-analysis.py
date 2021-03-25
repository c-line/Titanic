import numpy as np
import sklearn
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import math


train_data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')



nan_columns = test_data.columns[test_data.isna().any()].tolist()
nan_columns = nan_columns + ['Name','Ticket']
train_data = train_data.drop(nan_columns,1)
test_data = test_data.drop(nan_columns,1)

# drop any rows in the training data with an na
train_data = train_data.dropna(axis=0, how='any')

y = train_data["Survived"]
train_data = train_data.drop("Survived",1)

# print test_data.loc[152]

# print train_data.columns
# max_fare = train_data['Fare'].max()
# min_fare = train_data['Fare'].min()

# fare_bins = list(np.linspace(min_fare-1, max_fare+1, num=10, dtype=int))
# print train_data.loc[806]

# labels = [1,2,3,4,5,6,7,8,9]
# train_data['Fare'] = pd.cut(train_data['Fare'], fare_bins, labels=labels)
# test_data['Fare'] = pd.cut(test_data['Fare'], fare_bins, labels=labels)
# print train_data.to_string()
# train_data = train_data.astype({'Fare':'int64'})
# test_data = test_data.astype({'Fare':'int64'})
# print test_data.to_string()

# print train_data.dtypes
# print train_data.to_string()
# print train_data.loc[806]


X = pd.get_dummies(train_data)
X_test = pd.get_dummies(test_data)
print X.to_string()
# print list(train_data.columns)
# print list(test_data.columns)

# features = ["Pclass", "Sex", "SibSp", "Parch"]
# X = pd.get_dummies(train_data[features])
# X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
# print X.to_string()
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

