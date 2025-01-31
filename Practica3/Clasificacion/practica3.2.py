# Importamos las bibliotecas necesarias para el análisis
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Cargamos el archivo CSV con los datos de las transacciones
data = pd.read_csv('Transacciones_de_ventas Practicas.csv')

# Visualizamos las primeras filas del dataset para entender la estructura de los datos
print(data.head())
print("Columnas del dataset:", data.columns)
# --- PREPROCESAMIENTO DE LOS DATOS ---
# Selección de columnas relevantes para el modelo
# Usamos 'Edad', 'Sector' como variables independientes y 'Tipo' como variable objetivo
columns_to_use = ['Edad', 'Sector', 'Tipo']
data_cleaned = data[columns_to_use]  # Filtramos las columnas seleccionadas

# Codificamos las variables categóricas (si las hubiera) a valores numéricos usando LabelEncoder
label_encoders = {}
for column in data_cleaned.select_dtypes(include=['object']).columns:
    # Para cada columna categórica, convertimos las categorías en números
    label_encoders[column] = LabelEncoder()
    data_cleaned[column] = label_encoders[column].fit_transform(data_cleaned[column])

# Definimos las variables independientes (X) y la variable objetivo (y)
# Usamos solo 'Edad' y 'Sector' para visualizar el gráfico en 2D
X = data_cleaned[['Edad', 'Sector']]
y = data_cleaned['Tipo']

# Dividimos los datos en conjuntos de entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- MODELO SVM (Support Vector Machine) ---
# Creamos el modelo SVM con un kernel lineal
svm_model =  make_pipeline(StandardScaler(), SVC(random_state=42))
# Entrenamos el modelo SVM con los datos de entrenamiento
svm_model.fit(X_train, y_train)
# Realizamos predicciones con el modelo SVM sobre los datos de prueba
y_pred_svm = svm_model.predict(X_test)
precisión = accuracy_score(y_test, y_pred_svm)
print(f"Precisión del modelo: {precisión}")

# --- MODELO ÁRBOL DE DECISIÓN ---
# Creamos el modelo de Árbol de Decisión
tree_model = DecisionTreeClassifier(random_state=42)
# Entrenamos el modelo de Árbol de Decisión con los datos de entrenamiento
tree_model.fit(X_train, y_train)
# Realizamos predicciones con el modelo de Árbol de Decisión sobre los datos de prueba
y_pred_tree = tree_model.predict(X_test)

# --- EVALUACIÓN DE LOS MODELOS ---
# Matriz de confusión para el modelo SVM
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
# Matriz de confusión para el modelo de Árbol de Decisión
conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)

# --- GRAFICAMOS LAS MATRICES DE CONFUSIÓN ---
# Configuramos las figuras para ambas matrices de confusión
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Graficamos la matriz de confusión para el modelo SVM
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('SVM - Matriz de Confusión')
ax[0].set_xlabel('Predicción')
ax[0].set_ylabel('Valor Real')

# Graficamos la matriz de confusión para el modelo de Árbol de Decisión
sns.heatmap(conf_matrix_tree, annot=True, fmt='d', cmap='Greens', ax=ax[1])
ax[1].set_title('Árbol de Decisión - Matriz de Confusión')
ax[1].set_xlabel('Predicción')
ax[1].set_ylabel('Valor Real')

# Mostramos las gráficas de las matrices de confusión
plt.tight_layout()
plt.show()

# --- INFORMES DE CLASIFICACIÓN ---
# Imprimimos el informe de clasificación para el modelo SVM
print('Informe SVM:')
print(classification_report(y_test, y_pred_svm))

# Imprimimos el informe de clasificación para el modelo de Árbol de Decisión
print('Informe Árbol de Decisión:')
print(classification_report(y_test, y_pred_tree))

# --- GRAFICAMOS LAS FRONTERAS DE DECISIÓN ---
# Definimos los límites del gráfico en el espacio de 'Edad' y 'Sector'
x_min, x_max = X_train['Edad'].min() - 1, X_train['Edad'].max() + 1
y_min, y_max = X_train['Sector'].min() - 1, X_train['Sector'].max() + 1

# Creamos una malla de puntos para abarcar todo el espacio de características
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# --- GRÁFICO DE FRONTERA DE DECISIÓN PARA SVM ---
# Realizamos predicciones en cada punto de la malla con el modelo SVM
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficamos las fronteras de decisión para el modelo SVM
plt.figure(figsize=(10, 5))
plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')
plt.scatter(X_train['Edad'], X_train['Sector'], c=y_train, edgecolors='k', cmap='coolwarm')
plt.title('Frontera de Decisión - SVM')
plt.xlabel('Edad')
plt.ylabel('Sector')
plt.show()

# --- GRÁFICO DE FRONTERA DE DECISIÓN PARA ÁRBOL DE DECISIÓN ---
# Realizamos predicciones en cada punto de la malla con el modelo de Árbol de Decisión
Z = tree_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# --- GRÁFICO DE FRONTERA DE DECISIÓN PARA SVM ---
# Realizamos predicciones en cada punto de la malla con el modelo SVM
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficamos las fronteras de decisión para el modelo SVM
plt.figure(figsize=(10, 5))
plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')
plt.scatter(X_train['Edad'], X_train['Sector'], c=y_train, edgecolors='k', cmap='coolwarm')
plt.title('Frontera de Decisión - SVM')
plt.xlabel('Edad')
plt.ylabel('Sector')
plt.show()

# --- GRÁFICO DE FRONTERA DE DECISIÓN PARA ÁRBOL DE DECISIÓN ---
# Realizamos predicciones en cada punto de la malla con el modelo de Árbol de Decisión
Z = tree_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficamos las fronteras de decisión para el modelo de Árbol de Decisión
plt.figure(figsize=(10, 5))
plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')
plt.scatter(X_train['Edad'], X_train['Sector'], c=y_train, edgecolors='k', cmap='coolwarm')
plt.title('Frontera de Decisión - Árbol de Decisión')
plt.xlabel('Edad')
plt.ylabel('Sector')
plt.show()