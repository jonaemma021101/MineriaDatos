# Importamos las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

# Generamos datos simulados
np.random.seed(0)
n_samples = 3000

# Creamos el DataFrame
data = pd.DataFrame({
    'distancia_km': np.random.uniform(1, 50, n_samples),
    'trafico': np.random.uniform(0, 100, n_samples),
    'clima': np.random.choice([0, 1], n_samples),
    'hora_pico': np.random.choice([0, 1], n_samples),
    'tiempo_entrega': np.random.uniform(20, 200, n_samples)
})

# Ajustamos tiempo de entrega con retrasos
data['tiempo_entrega'] += (data['trafico'] * 0.3) + (data['clima'] * 15) + (data['hora_pico'] * 10)

# Gráficos exploratorios
plt.figure(figsize=(10, 5))
sns.heatmap(data.corr(), annot=True)
plt.title('Correlación de Variables')
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(x='distancia_km', y='tiempo_entrega', hue='trafico', palette='viridis', data=data)
plt.title('Relación Distancia vs Tiempo de Entrega')
plt.show()

# Modelo de Regresión Lineal
X = data[['distancia_km', 'trafico', 'clima', 'hora_pico']]
y = data['tiempo_entrega']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# Métricas del modelo
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Visualización de resultados
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([0, 250], [0, 250], color='red', linestyle='--')
plt.xlabel('Tiempo Real')
plt.ylabel('Tiempo Predicho')
plt.title('Predicción del Tiempo de Entrega')
plt.show()

# Clustering con K-means
kmeans = KMeans(n_clusters=3, random_state=0)
data['cluster'] = kmeans.fit_predict(data[['distancia_km', 'trafico']])

plt.figure(figsize=(10, 5))
sns.scatterplot(x='distancia_km', y='trafico', hue='cluster', palette='Set1', data=data)
plt.title('Agrupamiento de Rutas por Distancia y Tráfico')
plt.show()
