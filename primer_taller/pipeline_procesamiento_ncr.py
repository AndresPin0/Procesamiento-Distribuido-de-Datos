import pandas as pd
import numpy as np
from scipy import stats
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NCRDataPipeline:
    def __init__(self, input_file_path):
        self.input_file_path = input_file_path
        self.df = None
        self.df_clean = None
        self.output_dir = "output"
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_and_profile(self):
        print("=" * 60)
        print("1. CARGA Y PERFILADO RÁPIDO")
        print("=" * 60)
        
        try:
            self.df = pd.read_csv(self.input_file_path)
            print(f"Dataset cargado exitosamente")
            print(f"Forma del dataset: {self.df.shape}")
            print(f"Memoria utilizada: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            print("\nInformación del dataset:")
            print(self.df.info())
            
            print("\nPrimeras 5 filas:")
            print(self.df.head())
            
            print("\nEstadísticas descriptivas:")
            print(self.df.describe(include="all"))
            
            return True
            
        except Exception as e:
            print(f"Error al cargar el dataset: {e}")
            return False
    
    def standardize_data(self):
        print("\n" + "=" * 60)
        print("2. ESTANDARIZACIÓN DE DATOS")
        print("=" * 60)
        
        try:
            self.df.columns = self.df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
            print("Nombres de columnas normalizados a snake_case")
            
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['time'] = pd.to_datetime(self.df['time'], format='%H:%M:%S').dt.time
            print("Fechas y horas convertidas")
            
            numeric_columns = ['avg_vtat', 'avg_ctat', 'cancelled_rides_by_customer', 
                              'cancelled_rides_by_driver', 'incomplete_rides', 
                              'booking_value', 'ride_distance', 'driver_ratings', 'customer_rating']
            
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            print("Columnas numéricas convertidas")
            
            categorical_columns = ['booking_status', 'vehicle_type', 'pickup_location', 'drop_location',
                                  'reason_for_cancelling_by_customer', 'driver_cancellation_reason',
                                  'incomplete_rides_reason', 'payment_method']
            
            for col in categorical_columns:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype(str).str.lower().str.strip().str.replace(' ', '_')
            print("Columnas categóricas normalizadas")
            
            id_columns = ['booking_id', 'customer_id']
            for col in id_columns:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype(str).str.strip().str.replace('"', '')
            print("IDs limpiados")
            
            print(f"\nColumnas estandarizadas: {len(self.df.columns)}")
            print(f"Forma después de estandarización: {self.df.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error en estandarización: {e}")
            return False
    
    def handle_time_features(self):
        print("\n" + "=" * 60)
        print("3. MANEJO DE TIEMPO Y FEATURES")
        print("=" * 60)
        
        try:
            self.df['datetime'] = pd.to_datetime(self.df['date'].astype(str) + ' ' + self.df['time'].astype(str))
            print("Columna datetime creada")
            
            self.df['hour'] = self.df['datetime'].dt.hour
            self.df['day_of_week'] = self.df['datetime'].dt.day_name()
            self.df['month'] = self.df['datetime'].dt.month
            self.df['quarter'] = self.df['datetime'].dt.quarter
            print("Features de tiempo creados")
            
            print(f"\nDetección de duplicados:")
            print(f"   - Registros antes de eliminar duplicados: {len(self.df):,}")
            
            duplicates = self.df.duplicated(subset=['booking_id'], keep=False)
            print(f"   - Duplicados encontrados: {duplicates.sum():,}")
            
            self.df = self.df.drop_duplicates(subset=['booking_id'], keep='first')
            print(f"   - Registros después de eliminar duplicados: {len(self.df):,}")
            
            remaining_duplicates = self.df.duplicated(subset=['booking_id']).sum()
            print(f"   - Duplicados restantes: {remaining_duplicates}")
            
            print(f"\nForma después de manejo de tiempo: {self.df.shape}")
            return True
            
        except Exception as e:
            print(f"Error en manejo de tiempo: {e}")
            return False
    
    def clean_and_remove_outliers(self):
        print("\n" + "=" * 60)
        print("4. LIMPIEZA Y ELIMINACIÓN DE OUTLIERS")
        print("=" * 60)
        
        try:
            self.df_clean = self.df.copy()
            print(f"Dataset original: {len(self.df):,} registros")
            print(f"Dataset para limpieza: {len(self.df_clean):,} registros")
            
            def remove_outliers_iqr(df, column):
                if column not in df.columns or df[column].isna().all():
                    return df, None, None
                
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
                return df[mask], lower_bound, upper_bound
            
            numeric_cols = ['avg_vtat', 'avg_ctat', 'booking_value', 'ride_distance', 'driver_ratings', 'customer_rating']
            
            print(f"\nEliminación de outliers por columna:")
            total_removed = 0
            
            for col in numeric_cols:
                if col in self.df_clean.columns and self.df_clean[col].notna().sum() > 0:
                    initial_count = len(self.df_clean)
                    self.df_clean, lower, upper = remove_outliers_iqr(self.df_clean, col)
                    removed_count = initial_count - len(self.df_clean)
                    total_removed += removed_count
                    
                    if lower is not None and upper is not None:
                        print(f"   {col}:")
                        print(f"     - Límites: [{lower:.2f}, {upper:.2f}]")
                        print(f"     - Registros eliminados: {removed_count:,}")
                        print(f"     - Registros restantes: {len(self.df_clean):,}")
            
            print(f"\nResumen de limpieza:")
            print(f"   - Registros originales: {len(self.df):,}")
            print(f"   - Registros después de limpieza: {len(self.df_clean):,}")
            print(f"   - Total de registros eliminados: {total_removed:,}")
            print(f"   - Porcentaje de datos conservados: {(len(self.df_clean)/len(self.df)*100):.2f}%")
            
            return True
            
        except Exception as e:
            print(f"Error en limpieza: {e}")
            return False
    
    def calculate_business_metrics(self):
        print("\n" + "=" * 60)
        print("5. MÉTRICAS DEL NEGOCIO")
        print("=" * 60)
        
        try:
            def calculate_metrics_for_dataset(df, dataset_name):
                print(f"\n{dataset_name.upper()}")
                print("-" * 40)
                print(f"Registros en análisis: {len(df):,}")
                
                total_revenue = df['booking_value'].sum()
                print(f"Total de ingresos: ${total_revenue:,.2f}")
                
                avg_distance = df['ride_distance'].mean()
                print(f"Distancia promedio por viaje: {avg_distance:.2f} km")
                
                cancelled_bookings = df['booking_status'].str.contains('cancel', case=False, na=False).sum()
                total_bookings = len(df)
                cancellation_rate = (cancelled_bookings / total_bookings) * 100
                
                print(f"Tasa de cancelación: {cancellation_rate:.2f}%")
                print(f"   - Viajes cancelados: {cancelled_bookings:,}")
                print(f"   - Total de viajes: {total_bookings:,}")
                
                
                avg_revenue_per_ride = df['booking_value'].mean()
                print(f"   - Ingreso promedio por viaje: ${avg_revenue_per_ride:.2f}")
                
                total_distance = df['ride_distance'].sum()
                print(f"   - Distancia total recorrida: {total_distance:,.2f} km")
                
                avg_vtat = df['avg_vtat'].mean()
                avg_ctat = df['avg_ctat'].mean()
                print(f"   - Tiempo promedio VTAT: {avg_vtat:.2f} min")
                print(f"   - Tiempo promedio CTAT: {avg_ctat:.2f} min")
                
                avg_driver_rating = df['driver_ratings'].mean()
                avg_customer_rating = df['customer_rating'].mean()
                print(f"   - Calificación promedio conductores: {avg_driver_rating:.2f}/5")
                print(f"   - Calificación promedio clientes: {avg_customer_rating:.2f}/5")
                
                print(f"\nRESUMEN POR ESTADO DE RESERVA:")
                status_counts = df['booking_status'].value_counts()
                for status, count in status_counts.items():
                    percentage = (count / total_bookings) * 100
                    print(f"   - {status}: {count:,} ({percentage:.1f}%)")
            
            calculate_metrics_for_dataset(self.df, "DATASET ORIGINAL (CON OUTLIERS)")
            
            if self.df_clean is not None:
                calculate_metrics_for_dataset(self.df_clean, "DATASET LIMPIO (SIN OUTLIERS)")
                
                print(f"\nCOMPARACIÓN ENTRE DATASETS:")
                print("-" * 40)
                print(f"Registros originales: {len(self.df):,}")
                print(f"Registros después de limpieza: {len(self.df_clean):,}")
                print(f"Registros eliminados: {len(self.df) - len(self.df_clean):,}")
                print(f"Porcentaje de datos conservados: {(len(self.df_clean)/len(self.df)*100):.2f}%")
            
            return True
            
        except Exception as e:
            print(f"Error en cálculo de métricas: {e}")
            return False
    
    def save_processed_data(self):
        print("\n" + "=" * 60)
        print("6. GUARDADO DE DATASETS PROCESADOS")
        print("=" * 60)
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            output_path_final = os.path.join(self.output_dir, f"ncr_ride_bookings_processed_{timestamp}.csv")
            self.df.to_csv(output_path_final, index=False)
            print(f"Dataset procesado completo guardado en: {output_path_final}")
            
            if self.df_clean is not None:
                output_path_no_outliers = os.path.join(self.output_dir, f"ncr_ride_bookings_no_outliers_{timestamp}.csv")
                self.df_clean.to_csv(output_path_no_outliers, index=False)
                print(f"Dataset sin outliers guardado en: {output_path_no_outliers}")
            
            summary_path = os.path.join(self.output_dir, f"processing_summary_{timestamp}.txt")
            with open(summary_path, 'w') as f:
                f.write("RESUMEN DE PROCESAMIENTO NCR RIDE BOOKINGS\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Fecha de procesamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Archivo original: {self.input_file_path}\n")
                f.write(f"Registros originales: {len(self.df):,}\n")
                if self.df_clean is not None:
                    f.write(f"Registros después de limpieza: {len(self.df_clean):,}\n")
                    f.write(f"Registros eliminados: {len(self.df) - len(self.df_clean):,}\n")
                f.write(f"Columnas totales: {len(self.df.columns)}\n")
                f.write(f"Memoria utilizada: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
            
            print(f"Resumen de procesamiento guardado en: {summary_path}")
            
            print(f"\nArchivos guardados en el directorio '{self.output_dir}':")
            for file in os.listdir(self.output_dir):
                if file.endswith('.csv') or file.endswith('.txt'):
                    file_path = os.path.join(self.output_dir, file)
                    size = os.path.getsize(file_path) / 1024
                    print(f"   - {file}: {size:.1f} KB")
            
            return True
            
        except Exception as e:
            print(f"Error al guardar datos: {e}")
            return False
    
    def run_pipeline(self):
        print("INICIANDO PIPELINE DE PROCESAMIENTO NCR RIDE BOOKINGS")
        print("=" * 80)
        
        steps = [
            ("Carga y perfilado", self.load_and_profile),
            ("Estandarización", self.standardize_data),
            ("Manejo de tiempo", self.handle_time_features),
            ("Limpieza y outliers", self.clean_and_remove_outliers),
            ("Métricas del negocio", self.calculate_business_metrics),
            ("Guardado de datos", self.save_processed_data)
        ]
        
        for step_name, step_function in steps:
            print(f"\nEjecutando: {step_name}")
            if not step_function():
                print(f"Pipeline falló en: {step_name}")
                return False
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 80)
        return True

def main():
    input_file = "/Users/andres/Desktop/PDD/primer_taller/ncr_ride_bookings.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: No se encontró el archivo {input_file}")
        print("   Asegúrate de que el archivo esté en el directorio actual")
        return
    
    pipeline = NCRDataPipeline(input_file)
    success = pipeline.run_pipeline()
    
    if success:
        print("\nEl pipeline se ejecutó correctamente.")
        print("Revisa el directorio 'output' para los archivos procesados.")
    else:
        print("\nEl pipeline falló. Revisa los errores anteriores.")

if __name__ == "__main__":
    main()
