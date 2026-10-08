"""
Preprocesamiento de imágenes Sentinel-2 para el modelo Prithvi-EO-2.0.
Genera tiles 224x224 con solapamiento y normalización HLS.

Uso:
    python 02_preprocesar.py
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path

from config import (
    SOPO_BBOX, DATOS_DIR, TILES_DIR, SENTINEL_BANDS,
    TILE_SIZE, TILE_OVERLAP, HLS_MEAN, HLS_STD
)


def leer_bandas_tile(ruta_tile):
    """Lee todas las bandas de un tile descargado."""
    bandas = []
    for band in SENTINEL_BANDS:
        ruta = os.path.join(ruta_tile, f"{band}.tif")
        if os.path.exists(ruta):
            with rasterio.open(ruta) as src:
                data = src.read(1).astype(np.float32)
                bandas.append(data)
    return np.stack(bandas, axis=0) if bandas else None


def normalizar_hls(bandas):
    """
    Normaliza las bandas HLS a media 0 y std 1.
    bandas: numpy array (bands, H, W)
    """
    mean = np.array(HLS_MEAN, dtype=np.float32).reshape(-1, 1, 1)
    std = np.array(HLS_STD, dtype=np.float32).reshape(-1, 1, 1)

    # Escalar a rango 0-1 (Sentinel-2 L2A está en 0-10000)
    bandas = bandas / 10000.0
    bandas = np.clip(bandas, 0, 1)

    # Normalizar
    bandas = (bandas - mean) / std
    return bandas.astype(np.float32)


def generar_tiles(bandas, tamaño=224, solapamiento=0.5):
    """
    Genera tiles 224x224 con solapamiento.
    bandas: numpy array (bands, H, W)
    """
    step = int(tamaño * (1 - solapamiento))
    _, h, w = bandas.shape

    tiles = []
    coords = []

    for y in range(0, h - tamaño + 1, step):
        for x in range(0, w - tamaño + 1, step):
            tile = bandas[:, y:y+tamaño, x:x+tamaño]
            if tile.shape[1] == tamaño and tile.shape[2] == tamaño:
                tiles.append(tile)
                coords.append((y, x))

    return tiles, coords


def guardar_tile(tile, ruta_salida, transform=None, crs=None):
    """Guarda un tile como GeoTIFF."""
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    if transform is None:
        transform = rasterio.Affine.identity()

    if crs is None:
        crs = "EPSG:4326"

    with rasterio.open(
        ruta_salida,
        "w",
        driver="GTiff",
        height=tile.shape[1],
        width=tile.shape[2],
        count=tile.shape[0],
        dtype=tile.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(tile)


def preprocesar_imagenes():
    """
    Función principal: lee imágenes descargadas, normaliza y genera tiles.
    """
    print("=" * 60)
    print("PREPROCESAMIENTO DE IMÁGENES SENTINEL-2")
    print("=" * 60)

    # Buscar tiles descargados
    tiles_dirs = sorted(glob.glob(os.path.join(DATOS_DIR, "tile_*")))
    print(f"Encontrados {len(tiles_dirs)} tiles descargados")

    if not tiles_dirs:
        print("No hay imágenes descargadas. Ejecuta 01_descargar_imagenes.py primero.")
        return

    total_tiles = 0

    for tile_dir in tiles_dirs:
        nombre_tile = os.path.basename(tile_dir)
        print(f"\nProcesando {nombre_tile}...")

        # Leer bandas
        bandas = leer_bandas_tile(tile_dir)
        if bandas is None:
            print(f"  No se pudieron leer las bandas de {nombre_tile}")
            continue

        print(f"  Forma: {bandas.shape} (bands, H, W)")

        # Normalizar
        bandas_norm = normalizar_hls(bandas)

        # Generar tiles 224x224
        tiles, coords = generar_tiles(bandas_norm, TILE_SIZE, TILE_OVERLAP)
        print(f"  Generados {len(tiles)} tiles de {TILE_SIZE}x{TILE_SIZE}")

        # Guardar tiles
        for i, (tile, (y, x)) in enumerate(zip(tiles, coords)):
            ruta_salida = os.path.join(TILES_DIR, f"{nombre_tile}_tile_{i:04d}.npy")
            np.save(ruta_salida, tile)
            total_tiles += 1

    print(f"\n{'=' * 60}")
    print(f"Total de tiles generados: {total_tiles}")
    print(f"Directorio: {TILES_DIR}")
    print("=" * 60)


def crear_composicion_temporal():
    """
    Crea una composición temporal (mediana) de múltiples imágenes.
    Útil para reducir el efecto de nubes y sombras.
    """
    print("\nCreando composición temporal (mediana)...")

    tiles_dirs = sorted(glob.glob(os.path.join(DATOS_DIR, "tile_*")))
    if len(tiles_dirs) < 2:
        print("Se necesitan al menos 2 imágenes para composición temporal")
        return

    # Leer todas las bandas de cada tile
    todas_bandas = []
    for tile_dir in tiles_dirs:
        bandas = leer_bandas_tile(tile_dir)
        if bandas is not None:
            todas_bandas.append(bandas)

    if not todas_bandas:
        print("No se pudieron leer las bandas")
        return

    # Calcular mediana temporal
    todas_bandas = np.stack(todas_bandas, axis=0)  # (n_tiles, bands, H, W)
    composicion = np.median(todas_bandas, axis=0).astype(np.float32)

    # Normalizar
    composicion_norm = normalizar_hls(composicion)

    # Guardar composición
    ruta_salida = os.path.join(DATOS_DIR, "composicion_temporal.npy")
    np.save(ruta_salida, composicion_norm)
    print(f"Composición temporal guardada en: {ruta_salida}")

    # Generar tiles de la composición
    tiles, coords = generar_tiles(composicion_norm, TILE_SIZE, TILE_OVERLAP)
    for i, tile in enumerate(tiles):
        ruta_salida = os.path.join(TILES_DIR, f"composicion_tile_{i:04d}.npy")
        np.save(ruta_salida, tile)

    print(f"Generados {len(tiles)} tiles de la composición temporal")


if __name__ == "__main__":
    preprocesar_imagenes()
    crear_composicion_temporal()
