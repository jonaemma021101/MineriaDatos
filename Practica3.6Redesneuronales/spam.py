# Importar las librerías necesarias
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_20newsgroups

# Cargar un conjunto de datos de correos electrónicos (puedes usar otro dataset si prefieres)
data = fetch_20newsgroups(subset='all', categories=['sci.space', 'talk.politics.misc'], shuffle=True, random_state=42)

# Etiquetas: 0 es "no spam" y 1 es "spam" (en este caso usamos dos categorías generales)
emails = data.data
labels = data.target

# Convertir el texto a una matriz de cuentas de palabras
vectorizer = CountVectorizer(stop_words='english', max_features=1000)  # Se extraen las 1000 palabras más comunes
X = vectorizer.fit_transform(emails).toarray()

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Definir el modelo de red neuronal
model = Sequential([
    Dense(16, input_dim=1000, activation='relu'),  # Capa oculta con 16 neuronas y activación ReLU
    Dense(1, activation='sigmoid')  # Capa de salida con 1 neurona para la clasificación binaria (spam/no spam)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Evaluar el modelo en los datos de prueba
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Precisión en los datos de prueba: {accuracy:.4f}")
