"""
Descarga imágenes Sentinel-2 L2A desde AWS Open Data Registry.
Sin autenticación: los datos son públicos y accesibles vía HTTP.

Uso:
    python 01_descargar_imagenes.py
"""

import os
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path

import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

from config import (
    SOPO_BBOX, DATOS_DIR, SENTINEL_AWS_BUCKET, SENTINEL_AWS_REGION,
    SENTINEL_BANDS, TEMPORADA_SECA_MESES
)


def buscar_tiles_sentinel(bbox, fecha_inicio, fecha_fin, max_nubes=20):
    """
    Busca tiles Sentinel-2 L2A en AWS Open Data usando el catálogo STAC público.
    No requiere autenticación.
    """
    stac_url = "https://earth-search.aws.element84.com/v1"

    query = {
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{fecha_inicio}/{fecha_fin}",
        "bbox": [bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"]],
        "query": {
            "eo:cloud_cover": {"lt": max_nubes}
        },
        "limit": 10,
    }

    print(f"Buscando imágenes Sentinel-2 entre {fecha_inicio} y {fecha_fin}...")
    response = requests.post(f"{stac_url}/search", json=query, timeout=60)
    response.raise_for_status()
    data = response.json()

    features = data.get("features", [])
    print(f"Se encontraron {len(features)} imágenes con <{max_nubes}% nubes")

    return features


def descargar_band_href(href, ruta_salida):
    """Descarga un asset desde URL pública de AWS S3."""
    if os.path.exists(ruta_salida):
        print(f"  Ya existe: {os.path.basename(ruta_salida)}")
        return True

    print(f"  Descargando: {os.path.basename(ruta_salida)}...")
    try:
        response = requests.get(href, stream=True, timeout=300)
        response.raise_for_status()
        with open(ruta_salida, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  Error descargando: {e}")
        return False


def descargar_imagenes_sopo(fecha_inicio="2024-01-01", fecha_fin="2024-03-31"):
    """
    Función principal: descarga imágenes Sentinel-2 para Sopó.
    Usa temporada seca (enero-marzo 2024) para minimizar nubes.
    """
    print("=" * 60)
    print("DESCARGA DE IMÁGENES SENTINEL-2 PARA SOPÓ, CUNDINAMARCA")
    print("=" * 60)
    print(f"Bounding box: {SOPO_BBOX}")
    print(f"Periodo: {fecha_inicio} a {fecha_fin}")
    print(f"Resolución: 10m | Bandas: {SENTINEL_BANDS}")
    print("-" * 60)

    features = buscar_tiles_sentinel(SOPO_BBOX, fecha_inicio, fecha_fin)

    if not features:
        print("No se encontraron imágenes. Intenta con otro periodo.")
        return []

    imagenes_descargadas = []

    for i, feature in enumerate(features):
        print(f"\nImagen {i+1}/{len(features)}:")
        print(f"  Fecha: {feature['properties']['datetime']}")
        print(f"  Nubes: {feature['properties'].get('eo:cloud_cover', 'N/A')}%")

        tile_dir = os.path.join(DATOS_DIR, f"tile_{i:02d}")
        os.makedirs(tile_dir, exist_ok=True)

        bandas_ok = True
        for band in SENTINEL_BANDS:
            # Construir URL pública de AWS S3 para Sentinel-2 L2A
            # Formato: sentinel-s2-l2a/tiles/{utm}/{lat}/{band}/{year}/{month}/{day}/{scene}/{band}.tif
            assets = feature.get("assets", {})

            # Buscar la banda en los assets del STAC
            band_key = None
            for key in assets:
                if band.lower() in key.lower():
                    band_key = key
                    break

            if band_key is None:
                print(f"  Banda {band} no encontrada en assets")
                bandas_ok = False
                break

            href = assets[band_key]["href"]
            ruta_salida = os.path.join(tile_dir, f"{band}.tif")

            if not descargar_band_href(href, ruta_salida):
                bandas_ok = False
                break

        if bandas_ok:
            imagenes_descargadas.append(tile_dir)
            print(f"  Tile guardado en: {tile_dir}")

    print(f"\n{'=' * 60}")
    print(f"Total imágenes descargadas: {len(imagenes_descargadas)}")
    print(f"Directorio: {DATOS_DIR}")
    print("=" * 60)

    return imagenes_descargadas


def descargar_por_aws_directo():
    """
    Alternativa: descarga directa desde AWS S3 público.
    Usa el bucket sentinel-s2-l2a sin autenticación.
    """
    print("\nDescarga directa desde AWS S3 (sentinel-s2-l2a)...")

    # Para Sopó: UTM zone 18N, tile 18NTP
    # Path en AWS S3: tiles/18/N/TP/2024/1/15/
    base_s3 = f"https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs"

    # Tile UTM para Sopó
    utm_zone = 18
    lat_band = "N"
    grid_square = "TP"

    # Fechas de ejemplo (temporada seca 2024)
    fechas = ["2024/1/15", "2024/2/14", "2024/3/15"]

    for fecha in fechas:
        year, month, day = fecha.split("/")
        # Construir path del scene
        scene_path = f"{utm_zone}/{lat_band}/{grid_square}/{year}/{month}/{day}"

        print(f"\nProbando escena: {fecha}")
        # Nota: la estructura exacta del path puede variar
        # Se recomienda usar el catálogo STAC para obtener URLs exactas

    print("Usa buscar_tiles_sentinel() para URLs exactas del catálogo STAC.")


if __name__ == "__main__":
    # Descargar imágenes de temporada seca 2024
    descargar_imagenes_sopo("2024-01-01", "2024-03-31")
