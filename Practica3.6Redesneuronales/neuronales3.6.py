# Importar las librerías necesarias
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Cargar el conjunto de datos MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocesamiento: normalizar las imágenes (escala 0-1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convertir las etiquetas a formato one-hot
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Definir el modelo de red neuronal
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Aplanar las imágenes 28x28 en un vector de 784 elementos
    Dense(128, activation='relu'),  # Capa oculta con 128 neuronas y función de activación ReLU
    Dense(10, activation='softmax')  # Capa de salida con 10 neuronas (una por cada dígito)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_images, train_labels, epochs=5, batch_size=32)

# Evaluar el modelo en los datos de prueba
test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc_por= test_acc * 100
print(f"\nPrecisión en los datos de prueba: {test_acc_por:.4f}%\n")
