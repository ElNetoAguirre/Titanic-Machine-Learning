"""
Titanic - Machine Learning
"""

# importar bibliotecas
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree
# from sklearn.metrics.pairwise import normalize

# leer archivo csv "DataSet_Titanic.csv"
df = pd.read_csv("DataSet_Titanic.csv")

# guardar en variable X los atributos predictores (todas las etiquetas excepto "Sobreviviente")
X = df.drop("Sobreviviente", axis = 1)

# guardar en y la etiqueta a predecir ("Sobreviviente")
y = df.Sobreviviente

# Creamos un objeto arbol
arbol = DecisionTreeClassifier(max_depth = 4, random_state = 42)

# entrenamos a la máquina
arbol.fit(X, y)

# Predecimos sobre nuestro set
pred_y = arbol.predict(X)

# Comaparamos con las etiquetas reales
print("Precisión: ", accuracy_score(pred_y, y))

# creamos una matriz de confusión
confusion_matrix(y, pred_y)

# creamos un gráfico para la matriz de confusión
cm = confusion_matrix(y, pred_y)
disp = ConfusionMatrixDisplay(cm, display_labels = arbol.classes_)
disp.plot(cmap = plt.cm.Blues)

# creamos un gráfico para la matriz de confusión normalizada
ConfusionMatrixDisplay.from_estimator(arbol, X, y,
                                      cmap=plt.cm.Blues,
                                      values_format=".2f",
                                      normalize="true")
plt.show()

# mostramos un árbol gráficamente
plt.figure(figsize = (10, 8))
tree.plot_tree(arbol, filled = True, feature_names = X.columns)
plt.show()

# graficamos las importancias en un gráfico de barras
# creamos las variables x (importancias) e y (columnas)
importancias = arbol.feature_importances_
columnas = X.columns

# creamos el gráfico
sns.barplot(x = columnas, y = importancias)
plt.title("Importancia de cada atributo")
plt.show()
