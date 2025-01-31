# Importamos las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

# Cargamos el archivo CSV proporcionado
archivo = 'Transacciones_de_ventas Practicas.csv'
datos = pd.read_csv(archivo)

# Mostramos una vista previa de los datos
print("Vista previa de los datos:")
print(datos.head())

columns_to_use = ['Antiguedad', 'Edad', 'No._Hijos','Sexo','Estado_Civil']
data_cleaned = datos[columns_to_use]  # Filtramos las columnas seleccionadas

# Codificamos las variables categóricas (si las hubiera) a valores numéricos usando LabelEncoder
label_encoders = {}
for column in data_cleaned.select_dtypes(include=['object']).columns:
    # Para cada columna categórica, convertimos las categorías en números
    label_encoders[column] = LabelEncoder()
    data_cleaned[column] = label_encoders[column].fit_transform(data_cleaned[column])

# Definimos las variables independientes (X) y la variable objetivo (y)
# Usamos solo 'Edad' y 'Sector' para visualizar el gráfico en 2D
X = data_cleaned[['Antiguedad', 'Edad', 'No._Hijos','Sexo']]
y = data_cleaned['Estado_Civil']

columnas=X.columns
# Dividimos los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creación del modelo de árbol de decisión
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Predicciones sobre el conjunto de prueba
predicciones = modelo.predict(X_test)

# Resultados numéricos
print("\nReporte de clasificación:")
print(classification_report(y_test, predicciones))

print("Matriz de confusión:")
print(confusion_matrix(y_test, predicciones))

print("Precisión del modelo:")
print(accuracy_score(y_test, predicciones))

# Gráfico del árbol de decisión
plt.figure(figsize=(12,8))
tree.plot_tree(modelo, filled=True, feature_names=columnas,class_names=True)
plt.title("Árbol de decisión")
plt.show()

# Graficar algunas distribuciones de las variables independientes
for var in X:
    plt.figure(figsize=(6,4))
    sns.histplot(datos[var], kde=True)
    plt.title(f"Distribución de {var}")
    plt.show()

# Graficar la matriz de confusión
plt.figure(figsize=(6,6))
sns.heatmap(confusion_matrix(y_test, predicciones), annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de confusión")
plt.show()
