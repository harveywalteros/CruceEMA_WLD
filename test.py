import pandas as pd
import numpy as np
import talib
import ccxt
import websocket

print("🎉 ¡Todas las librerías funcionan!")
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("TA-Lib version:", talib.__version__)
print("CCXT version:", ccxt.__version__)
print("WebSocket disponible ✅")

# Crear un DataFrame de prueba
df = pd.DataFrame({
    'precio': [100, 102, 101, 103, 105]
})
print("\nDataFrame de prueba:")
print(df)