from efficient_probit_regression.datasets import Iris 

# Iris() ist ein objekt, daher die Klammern
data = Iris()

# die Funktion get_X() erzeugt aus dem Datensatz Iris() einen DataFrame X
# returned numpy.array mit Daten von X
print(data.get_X())
