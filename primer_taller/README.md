# Pipeline de Procesamiento NCR Ride Bookings

Pipeline completo para procesar y analizar el dataset de reservas de viajes NCR.

## Descripción

Este pipeline automatiza el procesamiento completo de datos de reservas de viajes, incluyendo estandarización, limpieza, eliminación de outliers y cálculo de métricas del negocio.

## Requisitos

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python pipeline_procesamiento_ncr.py
```

## Estructura del Pipeline

### 1. Carga y Perfilado Rápido
- Carga del dataset CSV
- Información básica (forma, tipos de datos, memoria)
- Estadísticas descriptivas iniciales

### 2. Estandarización de Datos
- Nombres de columnas a snake_case
- Conversión de tipos (fechas, numéricos)
- Normalización de categorías (minúsculas, sin espacios)
- Limpieza de IDs

### 3. Manejo de Tiempo y Features
- Unión de Date + Time → datetime
- Creación de features de tiempo (hora, día, mes, trimestre)
- Detección y eliminación de duplicados por booking_id

### 4. Limpieza y Eliminación de Outliers
- Detección de outliers usando IQR (1.5*IQR)
- Eliminación por columna numérica
- Creación de dataset limpio sin outliers

### 5. Métricas del Negocio
- **Dataset Original**: Métricas con outliers incluidos
- **Dataset Limpio**: Métricas sin outliers
- Comparación entre ambos datasets

### 6. Guardado de Datos
- Dataset procesado completo
- Dataset sin outliers
- Resumen de procesamiento
- Archivos con timestamp

## Archivos de Salida

- `ncr_ride_bookings_processed_[timestamp].csv` - Dataset procesado completo
- `ncr_ride_bookings_no_outliers_[timestamp].csv` - Dataset limpio sin outliers
- `processing_summary_[timestamp].txt` - Resumen del procesamiento

## Métricas Calculadas

- Total de ingresos
- Distancia promedio por viaje
- Tasa de cancelación
- Ingreso promedio por viaje
- Distancia total recorrida
- Tiempos promedio (VTAT, CTAT)
- Calificaciones promedio (conductores, clientes)
- Distribución por estado de reserva

## Archivos del Proyecto

- `pipeline_procesamiento_ncr.py` - Script principal del pipeline
- `procesamiento_ncr.ipynb` - Notebook de Jupyter con el mismo procesamiento paso a paso
- `requirements.txt` - Dependencias del proyecto
- `README.md` - Esta documentación

## Configuración

El pipeline busca automáticamente el archivo `ncr_ride_bookings.csv` en el directorio especificado. Modifica la variable `input_file` en la función `main()` si necesitas cambiar la ruta.

## Alternativas de Uso

**Pipeline Automatizado**: Ejecuta `python pipeline_procesamiento_ncr.py` para procesamiento completo automático.

**Notebook Interactivo**: Abre `procesamiento_ncr.ipynb` en Jupyter para procesamiento paso a paso con visualizaciones y análisis interactivo.
