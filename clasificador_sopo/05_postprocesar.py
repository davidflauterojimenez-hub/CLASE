"""
Postprocesamiento de predicciones: mosaico, estadísticas y exportación.
Genera GeoTIFF clasificado y estadísticas de cobertura.

Uso:
    python 05_postprocesar.py
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    SOPO_BBOX, RESULTADOS_DIR, MODELO_DIR, CLASES_COBERTURA, NUM_CLASES,
    TILE_SIZE, TILE_OVERLAP, CONFIDENCE_THRESHOLD
)


def crear_mosaico_predicciones():
    """
    Crea un mosaico con todas las predicciones de tiles.
    """
    print("=" * 60)
    print("POSTPROCESAMIENTO: MOSAICO DE PREDICCIONES")
    print("=" * 60)

    # Buscar predicciones
    pred_dir = os.path.join(MODELO_DIR, "predicciones")
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*_pred.npy")))

    if not pred_files:
        print("No se encontraron predicciones. Ejecuta 04_inferencia.py primero.")
        return None

    print(f"Predicciones encontradas: {len(pred_files)}")

    # Leer todas las predicciones
    predicciones = []
    for f in pred_files:
        pred = np.load(f)
        predicciones.append(pred)

    # Crear mosaico simple (asumiendo grid regular)
    # Para un mosaico más complejo, se necesitarían las coordenadas de cada tile
    n = len(predicciones)
    grid_size = int(np.ceil(np.sqrt(n)))

    mosaico = np.zeros((grid_size * TILE_SIZE, grid_size * TILE_SIZE), dtype=np.uint8)

    for i, pred in enumerate(predicciones):
        row = i // grid_size
        col = i % grid_size
        y_start = row * TILE_SIZE
        x_start = col * TILE_SIZE
        mosaico[y_start:y_start+TILE_SIZE, x_start:x_start+TILE_SIZE] = pred

    # Guardar mosaico
    ruta_mosaico = os.path.join(RESULTADOS_DIR, "mosaico_predicciones.npy")
    np.save(ruta_mosaico, mosaico)
    print(f"Mosaico guardado en: {ruta_mosaico}")
    print(f"Dimensiones: {mosaico.shape}")

    return mosaico


def calcular_estadisticas(mosaico):
    """
    Calcula estadísticas de cobertura terrestre.
    """
    print("\n" + "=" * 60)
    print("ESTADÍSTICAS DE COBERTURA TERRESTRE - SOPÓ")
    print("=" * 60)

    total_pixels = mosaico.size
    estadisticas = {}

    for clase_id, info in CLASES_COBERTURA.items():
        count = np.sum(mosaico == clase_id)
        porcentaje = (count / total_pixels) * 100
        estadisticas[clase_id] = {
            "nombre": info["nombre"],
            "pixeles": int(count),
            "porcentaje": round(porcentaje, 2),
            "color": info["color"]
        }
        print(f"  {info['nombre']:25s}: {porcentaje:6.2f}% ({count:,} pixels)")

    # Guardar estadísticas
    import json
    ruta_stats = os.path.join(RESULTADOS_DIR, "estadisticas_cobertura.json")
    with open(ruta_stats, "w", encoding="utf-8") as f:
        json.dump(estadisticas, f, indent=2, ensure_ascii=False)
    print(f"\nEstadísticas guardadas en: {ruta_stats}")

    return estadisticas


def exportar_geotiff(mosaico, ruta_salida=None):
    """
    Exporta el mosaico como GeoTIFF georreferenciado.
    """
    if ruta_salida is None:
        ruta_salida = os.path.join(RESULTADOS_DIR, "cobertura_sopo.tif")

    print(f"\nExportando GeoTIFF: {ruta_salida}")

    # Transformada geográfica para Sopó
    transform = from_bounds(
        SOPO_BBOX["lon_min"], SOPO_BBOX["lat_min"],
        SOPO_BBOX["lon_max"], SOPO_BBOX["lat_max"],
        mosaico.shape[1], mosaico.shape[0]
    )

    with rasterio.open(
        ruta_salida,
        "w",
        driver="GTiff",
        height=mosaico.shape[0],
        width=mosaico.shape[1],
        count=1,
        dtype=mosaico.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(mosaico, 1)
        dst.set_band_description(1, "Cobertura terrestre")

        # Añadir tabla de colores
        colormap = {}
        for clase_id, info in CLASES_COBERTURA.items():
            # Convertir hex a RGB
            hex_color = info["color"].lstrip("#")
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            colormap[clase_id] = rgb + (255,)
        dst.write_colormap(1, colormap)

    print(f"GeoTIFF exportado: {ruta_salida}")
    return ruta_salida


def generar_grafico_estadisticas(estadisticas, ruta_salida=None):
    """
    Genera un gráfico de barras con las coberturas.
    """
    if ruta_salida is None:
        ruta_salida = os.path.join(RESULTADOS_DIR, "grafico_coberturas.png")

    print(f"\nGenerando gráfico: {ruta_salida}")

    nombres = [info["nombre"] for info in estadisticas.values()]
    porcentajes = [info["porcentaje"] for info in estadisticas.values()]
    colores = [info["color"] for info in estadisticas.values()]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(nombres, porcentajes, color=colores, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Porcentaje (%)", fontsize=12)
    ax.set_ylabel("Tipo de cobertura", fontsize=12)
    ax.set_title("Cobertura Terrestre - Sopó, Cundinamarca", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(porcentajes) * 1.2)

    # Añadir valores en las barras
    for bar, pct in zip(bars, porcentajes):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Gráfico guardado: {ruta_salida}")
    return ruta_salida


def generar_shapefile(mosaico, ruta_salida=None):
    """
    Genera un shapefile con polígonos por clase de cobertura.
    """
    if ruta_salida is None:
        ruta_salida = os.path.join(RESULTADOS_DIR, "cobertura_sopo.shp")

    print(f"\nGenerando shapefile: {ruta_salida}")

    from rasterio import features
    from shapely.geometry import shape

    # Transformada geográfica
    transform = from_bounds(
        SOPO_BBOX["lon_min"], SOPO_BBOX["lat_min"],
        SOPO_BBOX["lon_max"], SOPO_BBOX["lat_max"],
        mosaico.shape[1], mosaico.shape[0]
    )

    # Extraer polígonos por clase
    shapes = []
    for clase_id, info in CLASES_COBERTURA.items():
        mask = (mosaico == clase_id).astype(np.uint8)
        if mask.sum() == 0:
            continue

        for geom, val in features.shapes(mask, mask=mask, transform=transform):
            if val == 1:
                shapes.append({
                    "geometry": shape(geom),
                    "clase_id": clase_id,
                    "nombre": info["nombre"],
                    "color": info["color"]
                })

    if shapes:
        gdf = gpd.GeoDataFrame(shapes, crs="EPSG:4326")
        gdf.to_file(ruta_salida)
        print(f"Shapefile guardado: {ruta_salida}")
        print(f"  {len(gdf)} polígonos")
    else:
        print("No se pudieron extraer polígonos")

    return ruta_salida


def postprocesar():
    """
    Función principal de postprocesamiento.
    """
    # Crear mosaico
    mosaico = crear_mosaico_predicciones()
    if mosaico is None:
        return

    # Estadísticas
    estadisticas = calcular_estadisticas(mosaico)

    # Exportar GeoTIFF
    exportar_geotiff(mosaico)

    # Gráfico
    generar_grafico_estadisticas(estadisticas)

    # Shapefile
    generar_shapefile(mosaico)

    print(f"\n{'=' * 60}")
    print("Postprocesamiento completado")
    print(f"Resultados en: {RESULTADOS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    postprocesar()
