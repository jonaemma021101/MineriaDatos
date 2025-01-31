# Importación de librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from scipy import stats

# 1. Cargar y preparar los datos
data = pd.read_csv("Ventas_Bebidas_Demo.csv", encoding='ISO-8859-1')

# Convertimos la columna 'FECHA' al tipo datetime para trabajar con fechas
data['FECHA'] = pd.to_datetime(data['FECHA'], format='%d-%m-%y')

# Agrupamos las ventas por TIENDA y FECHA, sumando los montos diarios
ventas_agrupadas = data.groupby(['TIENDA', 'FECHA'])['MONTO'].sum().reset_index()

# Seleccionar la tienda con mayor número de transacciones
tienda_top = ventas_agrupadas['TIENDA'].value_counts().idxmax()
serie_tienda = ventas_agrupadas[ventas_agrupadas['TIENDA'] == tienda_top]

# Crear la serie temporal diaria con índice de fechas y rellenar valores faltantes con 0
serie_tienda = serie_tienda.set_index('FECHA').asfreq('D').fillna(0)

# 2. Aplicar Triple Exponential Smoothing (TES)
# TES descompone la serie temporal en tendencia, estacionalidad y ciclo
model_tes = ExponentialSmoothing(
    serie_tienda['MONTO'], trend='add', seasonal='add', seasonal_periods=365
).fit()

# Pronóstico de ventas para los próximos 30 días
forecast_tes = model_tes.forecast(30)

# Calcular el MAPE (Error Absoluto Porcentual Medio) para evaluar el modelo
mape_tes = mean_absolute_percentage_error(serie_tienda['MONTO'], model_tes.fittedvalues)

# Graficar el ajuste y el pronóstico del modelo TES
plt.figure(figsize=(12, 6))
plt.plot(serie_tienda['MONTO'], label='Ventas reales', color='blue')
plt.plot(model_tes.fittedvalues, label='Ajuste TES', color='green')
plt.plot(forecast_tes, label='Pronóstico TES', color='red')
plt.legend(loc='best')
plt.title(f'TES - Ventas Reales vs Ajuste y Pronóstico (MAPE: {mape_tes:.2%})')
plt.show()

# 3. Descomposición de la serie temporal
# La descomposición muestra la tendencia, estacionalidad y los residuos
decomposition = seasonal_decompose(serie_tienda['MONTO'], model='additive', period=365)
decomposition.plot()
plt.show()

# 4. Ajustar el modelo ARIMA
# ACF y PACF nos ayudan a seleccionar los parámetros p, d y q del modelo ARIMA
lag_acf = acf(serie_tienda['MONTO'], nlags=20)
lag_pacf = pacf(serie_tienda['MONTO'], nlags=20)

# Graficar ACF y PACF para identificar los parámetros
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
axes[0].stem(lag_acf)
axes[0].set_title('ACF (Función de Autocorrelación)')
axes[1].stem(lag_pacf)
axes[1].set_title('PACF (Función de Autocorrelación Parcial)')
plt.show()

# Ajustar el modelo ARIMA con los parámetros p=1, d=1, q=1
model_arima = ARIMA(serie_tienda['MONTO'], order=(1, 1, 1)).fit()
forecast_arima = model_arima.forecast(30)

# Graficar el ajuste y el pronóstico del modelo ARIMA
plt.figure(figsize=(12, 6))
plt.plot(serie_tienda['MONTO'], label='Ventas reales', color='blue')
plt.plot(model_arima.fittedvalues, label='Ajuste ARIMA', color='green')
plt.plot(forecast_arima, label='Pronóstico ARIMA', color='red')
plt.legend(loc='best')
plt.title('ARIMA - Ventas Reales vs Ajuste y Pronóstico')
plt.show()

# 5. Prueba de Ruido Blanco (ADF Test)
# La prueba ADF verifica si la serie es estacionaria
adf_result = adfuller(serie_tienda['MONTO'])
print(f'Estadístico ADF: {adf_result[0]}')
print(f'Valor p: {adf_result[1]}')

if adf_result[1] < 0.05:
    print("La serie es estacionaria (rechazamos la hipótesis nula).")
else:
    print("La serie no es estacionaria (no rechazamos la hipótesis nula).")

# 6. Detección de Valores Atípicos (Outliers)
# Usamos el z-score para detectar outliers (valores fuera de ±3 desviaciones estándar)
z_scores = np.abs(stats.zscore(serie_tienda['MONTO']))
outliers = serie_tienda[z_scores > 3]

# Mostrar los valores atípicos detectados
print("Valores Atípicos Detectados:")
print(outliers)
