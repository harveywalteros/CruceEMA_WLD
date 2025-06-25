import pandas as pd
import numpy as np
import talib
import ccxt
from datetime import datetime
import time

class EstrategiaCrucesEMAs:
    def __init__(self, symbol='WLD/USDT', timeframe='1h', ema_rapida=12, ema_lenta=26):
        """
        Estrategia de cruces de EMAs usando TA-Lib
        
        Args:
            symbol (str): Par de trading (ej: 'WLD/USDT')
            timeframe (str): Marco temporal ('1m', '5m', '15m', '1h', '4h', '1d')
            ema_rapida (int): Período de EMA rápida
            ema_lenta (int): Período de EMA lenta
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.ema_rapida = ema_rapida
        self.ema_lenta = ema_lenta
        self.exchange = ccxt.binance()
        
    def obtener_datos(self, limit=100):
        """
        Obtiene datos OHLCV del exchange
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error obteniendo datos: {e}")
            return None
    
    def calcular_indicadores(self, df):
        """
        Calcula EMAs, VWAP y detecta cruces usando TA-Lib
        """
        # Calcular EMAs con TA-Lib
        df['ema_rapida'] = talib.EMA(df['close'].values, timeperiod=self.ema_rapida)
        df['ema_lenta'] = talib.EMA(df['close'].values, timeperiod=self.ema_lenta)
        
        # Calcular VWAP manualmente (TA-Lib no tiene VWAP)
        df['vwap'] = self.calcular_vwap(df)
        
        # Calcular diferencia entre EMAs
        df['diferencia_emas'] = df['ema_rapida'] - df['ema_lenta']
        
        # Calcular relación precio-VWAP
        df['precio_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        
        # Detectar cruces de EMAs
        df['cruce_alcista_emas'] = (
            (df['ema_rapida'] > df['ema_lenta']) & 
            (df['ema_rapida'].shift(1) <= df['ema_lenta'].shift(1))
        )
        
        df['cruce_bajista_emas'] = (
            (df['ema_rapida'] < df['ema_lenta']) & 
            (df['ema_rapida'].shift(1) >= df['ema_lenta'].shift(1))
        )
        
        # Detectar cruces de precio con VWAP
        df['precio_sobre_vwap'] = df['close'] > df['vwap']
        df['cruce_alcista_vwap'] = (
            (df['close'] > df['vwap']) & 
            (df['close'].shift(1) <= df['vwap'].shift(1))
        )
        
        df['cruce_bajista_vwap'] = (
            (df['close'] < df['vwap']) & 
            (df['close'].shift(1) >= df['vwap'].shift(1))
        )
        
        # Señales combinadas EMAs + VWAP
        df['cruce_alcista'] = df['cruce_alcista_emas'] & df['precio_sobre_vwap']
        df['cruce_bajista'] = df['cruce_bajista_emas'] & ~df['precio_sobre_vwap']
        
        # Determinar tendencia
        df['tendencia'] = np.where(
            (df['ema_rapida'] > df['ema_lenta']) & (df['close'] > df['vwap']), 'ALCISTA', 
            np.where((df['ema_rapida'] < df['ema_lenta']) & (df['close'] < df['vwap']), 'BAJISTA', 'NEUTRAL')
        )
        
        # Calcular RSI para confirmación
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
        
        # Calcular MACD para confirmación adicional
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'].values)
        
        return df
    
    def calcular_vwap(self, df):
        """
        Calcula VWAP (Volume Weighted Average Price)
        """
        # Precio típico (HLC/3)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # VWAP acumulativo
        vwap_values = []
        cumulative_pv = 0
        cumulative_volume = 0
        
        for i in range(len(df)):
            pv = typical_price.iloc[i] * df['volume'].iloc[i]
            cumulative_pv += pv
            cumulative_volume += df['volume'].iloc[i]
            
            if cumulative_volume > 0:
                vwap = cumulative_pv / cumulative_volume
            else:
                vwap = typical_price.iloc[i]
            
            vwap_values.append(vwap)
        
        return pd.Series(vwap_values, index=df.index)
    
    def generar_señales(self, df):
        """
        Genera señales de compra/venta basadas en cruces de EMAs
        """
        señales = []
        
        for i in range(len(df)):
            if df.iloc[i]['cruce_alcista']:
                # Señal de compra
                señal = {
                    'tipo': 'COMPRA',
                    'datetime': df.iloc[i]['datetime'],
                    'precio': df.iloc[i]['close'],
                    'ema_rapida': df.iloc[i]['ema_rapida'],
                    'ema_lenta': df.iloc[i]['ema_lenta'],
                    'vwap': df.iloc[i]['vwap'],
                    'precio_vs_vwap': df.iloc[i]['precio_vs_vwap'],
                    'rsi': df.iloc[i]['rsi'],
                    'macd': df.iloc[i]['macd'],
                    'confirmacion': self.confirmar_señal_compra(df.iloc[i])
                }
                señales.append(señal)
                
            elif df.iloc[i]['cruce_bajista']:
                # Señal de venta
                señal = {
                    'tipo': 'VENTA',
                    'datetime': df.iloc[i]['datetime'],
                    'precio': df.iloc[i]['close'],
                    'ema_rapida': df.iloc[i]['ema_rapida'],
                    'ema_lenta': df.iloc[i]['ema_lenta'],
                    'vwap': df.iloc[i]['vwap'],
                    'precio_vs_vwap': df.iloc[i]['precio_vs_vwap'],
                    'rsi': df.iloc[i]['rsi'],
                    'macd': df.iloc[i]['macd'],
                    'confirmacion': self.confirmar_señal_venta(df.iloc[i])
                }
                señales.append(señal)
        
        return señales
    
    def confirmar_señal_compra(self, fila):
        """
        Confirma señal de compra con RSI y MACD
        """
        confirmaciones = []
        
        # RSI no debe estar en sobrecompra
        if fila['rsi'] < 70:
            confirmaciones.append("RSI_OK")
        
        # MACD debe estar por encima de la línea de señal
        if fila['macd'] > fila['macd_signal']:
            confirmaciones.append("MACD_OK")
        
        return confirmaciones
    
    def confirmar_señal_venta(self, fila):
        """
        Confirma señal de venta con RSI y MACD
        """
        confirmaciones = []
        
        # RSI no debe estar en sobreventa
        if fila['rsi'] > 30:
            confirmaciones.append("RSI_OK")
        
        # MACD debe estar por debajo de la línea de señal
        if fila['macd'] < fila['macd_signal']:
            confirmaciones.append("MACD_OK")
        
        return confirmaciones
    
    def calcular_rendimiento(self, df, señales):
        """
        Calcula rendimiento de la estrategia
        """
        if len(señales) < 2:
            return None
        
        operaciones = []
        posicion = None
        
        for señal in señales:
            if señal['tipo'] == 'COMPRA' and posicion is None:
                posicion = {
                    'entrada': señal['precio'],
                    'fecha_entrada': señal['datetime']
                }
            elif señal['tipo'] == 'VENTA' and posicion is not None:
                rendimiento = (señal['precio'] - posicion['entrada']) / posicion['entrada'] * 100
                operaciones.append({
                    'entrada': posicion['entrada'],
                    'salida': señal['precio'],
                    'fecha_entrada': posicion['fecha_entrada'],
                    'fecha_salida': señal['datetime'],
                    'rendimiento': rendimiento
                })
                posicion = None
        
        if operaciones:
            rendimiento_total = sum([op['rendimiento'] for op in operaciones])
            operaciones_ganadoras = len([op for op in operaciones if op['rendimiento'] > 0])
            tasa_acierto = operaciones_ganadoras / len(operaciones) * 100
            
            return {
                'operaciones': operaciones,
                'rendimiento_total': rendimiento_total,
                'num_operaciones': len(operaciones),
                'tasa_acierto': tasa_acierto,
                'rendimiento_promedio': rendimiento_total / len(operaciones)
            }
        
        return None
    
    def ejecutar_estrategia(self):
        """
        Ejecuta la estrategia completa
        """
        print(f"🚀 Ejecutando estrategia para {self.symbol}")
        print(f"📊 Timeframe: {self.timeframe}")
        print(f"📈 EMA Rápida: {self.ema_rapida}, EMA Lenta: {self.ema_lenta}")
        print("-" * 50)
        
        # Obtener datos
        df = self.obtener_datos(limit=200)
        if df is None:
            return
        
        # Calcular indicadores
        df = self.calcular_indicadores(df)
        
        # Generar señales
        señales = self.generar_señales(df)
        
        # Mostrar últimas señales
        print("🎯 ÚLTIMAS SEÑALES:")
        for señal in señales[-5:]:  # Últimas 5 señales
            print(f"  {señal['tipo']} - {señal['datetime']} - Precio: ${señal['precio']:.4f}")
            print(f"    EMAs: {señal['ema_rapida']:.4f} / {señal['ema_lenta']:.4f}")
            print(f"    RSI: {señal['rsi']:.2f} - Confirmaciones: {señal['confirmacion']}")
            print()
        
        # Calcular rendimiento
        rendimiento = self.calcular_rendimiento(df, señales)
        if rendimiento:
            print("💰 RENDIMIENTO DE LA ESTRATEGIA:")
            print(f"  Rendimiento Total: {rendimiento['rendimiento_total']:.2f}%")
            print(f"  Número de Operaciones: {rendimiento['num_operaciones']}")
            print(f"  Tasa de Acierto: {rendimiento['tasa_acierto']:.2f}%")
            print(f"  Rendimiento Promedio: {rendimiento['rendimiento_promedio']:.2f}%")
        
        # Estado actual
        ultimo_row = df.iloc[-1]
        print("\n📊 ESTADO ACTUAL:")
        print(f"  Precio: ${ultimo_row['close']:.4f}")
        print(f"  EMA {self.ema_rapida}: {ultimo_row['ema_rapida']:.4f}")
        print(f"  EMA {self.ema_lenta}: {ultimo_row['ema_lenta']:.4f}")
        print(f"  Tendencia: {ultimo_row['tendencia']}")
        print(f"  RSI: {ultimo_row['rsi']:.2f}")
        
        return df, señales, rendimiento

def main():
    """
    Función principal
    """
    # Crear instancia de la estrategia
    estrategia = EstrategiaCrucesEMAs(
        symbol='WLD/USDT',
        timeframe='1h',
        ema_rapida=12,
        ema_lenta=26
    )
    
    # Ejecutar estrategia
    df, señales, rendimiento = estrategia.ejecutar_estrategia()
    
    # Puedes guardar los datos si quieres
    if df is not None:
        df.to_csv('datos_wld_estrategia.csv', index=False)
        print("\n💾 Datos guardados en 'datos_wld_estrategia.csv'")

if __name__ == "__main__":
    main()