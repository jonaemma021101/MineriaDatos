import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import matplotlib.pyplot as plt

# Cargar el archivo CSV
data = pd.read_csv('Ventas_Bebidas_Demo.csv', encoding='ISO-8859-1')

# Preprocesar los datos: crear una tabla de crosstab (pedido por sabor y marca)
basket = pd.crosstab(data['PEDIDO'], [data['SABOR'], data['MARCA']])

# Convertir los valores mayores a 0 en 1 (presencia del producto)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Definir el umbral mínimo de soporte
min_support = 0.05

# Aplicar el algoritmo Apriori
frequent_itemsets_apriori = apriori(basket, min_support=min_support, use_colnames=True)

# Aplicar el algoritmo FP-Growth
frequent_itemsets_fpgrowth = fpgrowth(basket, min_support=min_support, use_colnames=True)

# Obtener las reglas de asociación para Apriori
rules_apriori = association_rules(frequent_itemsets_apriori, metric="lift", min_threshold=1)

# Obtener las reglas de asociación para FP-Growth
rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="lift", min_threshold=1)

# Graficar los resultados para Apriori
plt.figure(figsize=(10, 6))
plt.scatter(rules_apriori['confidence'], rules_apriori['lift'], alpha=0.6, marker="o")
plt.title('Apriori - Confidence vs Lift')
plt.xlabel('Confidence')
plt.ylabel('Lift')
plt.grid(True)
plt.show()

# Graficar los resultados para FP-Growth
plt.figure(figsize=(10, 6))
plt.scatter(rules_fpgrowth['confidence'], rules_fpgrowth['lift'], alpha=0.6, marker="o", color='r')
plt.title('FP-Growth - Confidence vs Lift')
plt.xlabel('Confidence')
plt.ylabel('Lift')
plt.grid(True)
plt.show()

# Mostrar las reglas generadas por Apriori
print("Reglas generadas por Apriori:")
print(rules_apriori)

# Mostrar las reglas generadas por FP-Growth
print("\nReglas generadas por FP-Growth:")
print(rules_fpgrowth)



