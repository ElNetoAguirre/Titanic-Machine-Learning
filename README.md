<p align="center">
  <a href="https://www.python.org/" target="blank"><img src="https://www.pngmart.com/files/7/Python-PNG-Image.png" width="200" alt="Python Logo"/></a>
</p>

# Machine Learning

El programa revisa la base de datos de los pasajeros del Titanic, realiza un análisis de los sobrevivientes, identificando que carácteristicas son las más favorables para sobrevivir al hundimiento del barco.

A demás, si agregamos nuevos pasajeros, puede predecir si éstos sobreviviran o no al hundimiento.

Para la ejecución de las predicciones se utiliza un "Árbol de Deciciones", la cual es una de las tareas más importantes dentro de Machine Learning (dentro, a su vez, de lo que llamamos Aprendizaje Supervisado).

Clasificación en Machine Learning consiste en aprender etiquetas discretas "y" a partir de un conjunto de features "X" (que pueden ser uno, dos, o muchos más) tomando como muestra un conjunto de instancias.

El dataset de Titanic ha surgido de una competencia en el sitio Kaggle: Machine Learning from Disaster (https://www.kaggle.com/c/titanic).

El Dataset está compuesto por una serie de columnas, que tienen los siguientes significados:

1. **Sobreviviente**: 0 = No; 1 = Si
2. **Clase**: 1 = Primera Clase; 2 = Segunda Clase; 3 = Tercera Clase
3. **Género**: 0 = Hombre; 1 = Mujer
4. **Edad**: edad en años
5. **HermEsp**: cantidad de hermanos o esposos a bordo del Titanic, para el pasajero en cuestión
6. **PadHij**: cantidad de padres o hijos a bordo del Titanic, para el pasajero en cuestión

Para la generación del Árbol de decisión: como primera aproximación, diremos que es un objeto
que, dadas varias instancias con un determinado grupo de features `X` y unas determinadas etiquetas objetivo `y`, el árbol de desición aprende automáticamente reglas (de mayor a menor importancia) sobre
cada feature de manera de poder decidir qué etiqueta le corresponde a cada instancia.

Vamos a separar el dataset de Titanic en una variable `X` los atributos que usarás para predecir,
y en una variable `y` la etiqueta que quieres predecir. En este caso, si sobrevivió o no.

Si queremos entrenar un árbol de decisión para clasificar nuestras instancias, primero
debemos crear un objeto correspondiente al modelo. Este objeto será de de la clase
DecisionTreeClassifier, la cual importamos desde la librería Scikit-Learn.

Una vez que el modelo ha sido creado, debemos entrenarlo sobre nuestros datos. Esto
lo logramos con el método **"fit"** que poseen **"todas"** las clases correspondientes
a modelos de Scikit-Learn.

Ahora con una herramienta que, dadas ciertas características de una instancia, nos devuelve
qué etiqueta `y` que el modelo cree que le corresponde. Esto lo podemos hacer utilizando el
método **"predict"**, que también poseen **"todas"** las clases correspondientes a modelos de **"Scikit-Learn"**.

Para calcular el porcentaje de instancias bien clasificadas por el modelo? usamos nuevamente
el método **predict** sobre todo el dataset `X`. Luego con la función `accuracy_score` podemos
calcular el porcentaje de aciertos que obtenemos al comparar nuestra predicción `y_pred` contra
la clase original `y`, lo que nos devuelve un porcentaje de aciertos.

Esto quiere decir que se asigna la etiqueta correcta en el 80.25% de los casos.

Otra forma de ver los resultados de nuestro clasificador es la matriz de confusión. La matriz
de confusión es una tabla de doble entrada, donde un eje corresponde a la etiqueta real (y) y
otro a la etiqueta predicha (pred_y). En la diagonal encontramos los aciertos, mientras que por
fuera de la diagonal aquellas instancias mal clasificadas.

Al obtener el Árbol de Desición, la rama de la izquierda representa el resultado verdadero (True),
mientras que la rama derecha, representa el resultado falso (False).

El color de cada rectángulo representa la etiqueta predicha por el modelo (en éste caso, la
etiqueta azul representa el "sobrevive", o un valor de "y=1", y la naranja, "no sobrevive",
o un valor de "y=0").

A su vez, la tonalidad representa la seguridad que tiene el modelo en su predicción. A partir
del entrenamiento, el modelo aprendió algunas reglas para clasificar las instancias de acuerdo
a los valores asumidos por ciertas características. Dicha clasificación, sin embargo, contiene
errores, dado que esta división puede generar que una proporción (cuanto menor, mejor), de
instancias sean incorrectamente clasificadas, ya que en la realidad pertenecen a la otra
categoría. La cantidad de instancias incorrectamente clasficadas se procesa matemáticamente en
un indicador conocido como "impureza de Gini", el cual mide la cantidad de instacias incorrectamente
clasificadas dentro de cada "hoja" del árbol. Alcanza el valor mínimo de cero cuando no hay
instancias incorrectamente clasificadas. Esta información existe en nuestro gráfico de árbol,
y a su vez determina el color de la hoja, siendo más intenso cuando menor es el valor de la
"impureza de Gini", significando que la clasificación de esa hoja es más robusta para predecir
correctamente un resultado.

Éste modelo ha aprendido algunas cosas muy interesantes:
- La primera pregunta que el modelo hace, es acerca del sexo de la persona: si es
  hombre (0) a continuación se pregunta su edad. Si es un hombre de edad 7 años o más,
  le asigna una etiqueta de "no sobrevive". Por el contrario, si es un niño de 6 años o
      menos, predecirá "sobrevive".
- El caso es diferente si como resultado de la primera pregunta, el valor de sexo fuera
  1 (mujer). La pregunta que se hará a continuación el árbol es referido a qué clase
  pertenecía la pasajera: si fuera de 1° o 2° clase, le asignará la predicción "sobrevive",
  y si fuera de 3° clase, "no sobrevive".

En otras palabras, el modelo ha aprendido de los datos reales, literalmente nos ha dicho que
**"MUJERES Y NIÑOS PRIMERO"**. El modelo ha detectado que las mujeres tuvieron mayores oportunidades
de supervivencia (y cuanto mejor posición ecónomica, mejores serían sus oportunidades), y que en
el caso de los hombres, los niños pequeños tuvieron más suerte que los adolescentes o adultos.

Otra forma de visualizarlo graficando las importancias que han tenido cada una de las variables
en la predicción obtenida. Esta importancia es dada por Scikit-Learn a cada feature (x) en
función de qué tan útil ha sido para clasificar las instancias.

Podemos concluir entonces que, el factor más determinante fue el género, seguido de la clase
del pasajero, y luego la edad. Complementado con el diagrama anterior, pudimos ver cómo las
variables se influyeron mutuamente para determinar la posibilidad de supervivencia de acuerdo
al género de la persona.

---
Para la realización de éste código se usaron las librerías "numpy", "pandas" y "matplotlib", "seaborn".

---

## Glosario

**Machine Learning** = rama de la inteligencia artificial que, a través de algoritmos, dota a la computadora de la capacidad de identificar patrones en datos masivos y elaborar predicciones (análisis predictivo).

**numpy** = módulo por defecto para hacer operaciones numéricas de manera eficiente, incluso con cantidades extremadamente grandes de datos y en muy poco tiempo.

**pandas** = librería utilizada para manipular y analizar datos en Python, está montada sobre numpy, por lo cual muchas funcionalidades son similares. Tambien se le conoce como el Ecxel de Python por tener datos estructurados en filas y columnas (datasets).

**matplotlib** = la librería preferida para crear gráficos en Python.

**scikit-learn** = conjunto de módulos de python para aprendizaje automático y minería de datos

**seaborn** = visualización de datos estadísticos
