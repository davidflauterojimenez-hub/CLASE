"""
Pipeline principal: Clasificación de cobertura terrestre en Sopó, Cundinamarca.
Sin autenticación: usa datos públicos de AWS Open Data y modelos públicos de Hugging Face.

Uso:
    python main.py                    # Pipeline completo
    python main.py --paso 1           # Solo descargar imágenes
    python main.py --paso 2           # Solo preprocesar
    python main.py --paso 3           # Solo entrenar
    python main.py --paso 4           # Solo inferencia
    python main.py --paso 5           # Solo postprocesar
    python main.py --paso 6           # Solo visualizar
"""

import argparse
import sys
import time
from datetime import datetime

from config import SOPO_CENTRO, SOPO_BBOX


def mostrar_banner():
    """Muestra el banner del programa."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   CLASIFICADOR DE COBERTURA TERRESTRE                       ║
║   Sopó, Cundinamarca - Colombia                              ║
║                                                              ║
║   Modelo: Prithvi-EO-2.0-300M (NASA/IBM)                    ║
║   Datos: Sentinel-2 L2A (AWS Open Data)                      ║
║   Sin autenticación requerida                                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    print(f"Zona de estudio: Sopó, Cundinamarca")
    print(f"Coordenadas: {SOPO_CENTRO[0]}°N, {SOPO_CENTRO[1]}°W")
    print(f"Bounding box: lat [{SOPO_BBOX['lat_min']}, {SOPO_BBOX['lat_max']}], "
          f"lon [{SOPO_BBOX['lon_min']}, {SOPO_BBOX['lon_max']}]")
    print()


def paso_1_descargar():
    """Paso 1: Descargar imágenes Sentinel-2."""
    print("\n" + "=" * 60)
    print("PASO 1: Descarga de imágenes Sentinel-2")
    print("=" * 60)
    from 01_descargar_imagenes import descargar_imagenes_sopo
    return descargar_imagenes_sopo()


def paso_2_preprocesar():
    """Paso 2: Preprocesar imágenes."""
    print("\n" + "=" * 60)
    print("PASO 2: Preprocesamiento")
    print("=" * 60)
    from 02_preprocesar import preprocesar_imagenes, crear_composicion_temporal
    preprocesar_imagenes()
    crear_composicion_temporal()


def paso_3_entrenar():
    """Paso 3: Entrenar modelo."""
    print("\n" + "=" * 60)
    print("PASO 3: Entrenamiento del modelo")
    print("=" * 60)
    from 03_entrenar import entrenar
    entrenar()


def paso_4_inferencia():
    """Paso 4: Inferencia."""
    print("\n" + "=" * 60)
    print("PASO 4: Inferencia")
    print("=" * 60)
    from 04_inferencia import inferir_todos_tiles
    inferir_todos_tiles()


def paso_5_postprocesar():
    """Paso 5: Postprocesamiento."""
    print("\n" + "=" * 60)
    print("PASO 5: Postprocesamiento")
    print("=" * 60)
    from 05_postprocesar import postprocesar
    postprocesar()


def paso_6_visualizar():
    """Paso 6: Visualización."""
    print("\n" + "=" * 60)
    print("PASO 6: Visualización")
    print("=" * 60)
    from 06_visualizar import visualizar
    visualizar()


def main():
    """Función principal del pipeline."""
    parser = argparse.ArgumentParser(
        description="Clasificación de cobertura terrestre en Sopó, Cundinamarca"
    )
    parser.add_argument(
        "--paso", type=int, choices=range(1, 7), default=None,
        help="Ejecutar solo un paso específico (1-6). Si no se especifica, ejecuta todo."
    )
    parser.add_argument(
        "--fecha-inicio", type=str, default="2024-01-01",
        help="Fecha inicio para descarga (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--fecha-fin", type=str, default="2024-03-31",
        help="Fecha fin para descarga (YYYY-MM-DD)"
    )

    args = parser.parse_args()

    mostrar_banner()

    inicio = time.time()

    pasos = {
        1: paso_1_descargar,
        2: paso_2_preprocesar,
        3: paso_3_entrenar,
        4: paso_4_inferencia,
        5: paso_5_postprocesar,
        6: paso_6_visualizar,
    }

    if args.paso:
        # Ejecutar solo un paso
        pasos[args.paso]()
    else:
        # Ejecutar pipeline completo
        print("Ejecutando pipeline completo...")
        print()

        paso_1_descargar()
        paso_2_preprocesar()
        paso_3_entrenar()
        paso_4_inferencia()
        paso_5_postprocesar()
        paso_6_visualizar()

    fin = time.time()
    tiempo_total = fin - inicio

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO")
    print("=" * 60)
    print(f"Tiempo total: {tiempo_total:.1f} segundos ({tiempo_total/60:.1f} minutos)")
    print(f"Resultados en: resultados/")
    print("=" * 60)


if __name__ == "__main__":
    main()
