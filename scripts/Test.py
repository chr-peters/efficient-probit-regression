from efficient_probit_regression.datasets import Iris 

# Iris() ist ein objekt, daher die Klammern
data = Iris()

data
# die Funktion get_X() erzeugt aus dem Datensatz Iris() einen DataFrame X
# returned numpy.array mit Daten von X
# print(data.get_X())

# perform logistic regression on Iris dataset

from sklearn import linear_model
from sklearn.model_selection import train_test_split


target = data.get_y()

feature = data.get_X()

# train test split
feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.3, random_state=42)

# instancing unregularized logistic Regression 

logreg = linear_model.LogisticRegression(penalty = "none")

# fitting a logistic regression model

logreg.fit(X = feature_train, y = target_train)

# computing accuracy on the test data
accuracy = logreg.score(X = feature_test, y = target_test)

print(accuracy)
# 0.7777777777777778