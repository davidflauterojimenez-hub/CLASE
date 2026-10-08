"""
Configuración del clasificador de cobertura terrestre para Sopó, Cundinamarca.
Sin autenticación: usa datos públicos de AWS Open Data y modelos públicos de Hugging Face.
"""

import os

# ─── Ruta base del proyecto ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATOS_DIR = os.path.join(BASE_DIR, "datos")
TILES_DIR = os.path.join(BASE_DIR, "tiles")
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados")
MODELO_DIR = os.path.join(BASE_DIR, "modelo")

for d in [DATOS_DIR, TILES_DIR, RESULTADOS_DIR, MODELO_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Zona de estudio: Sopó, Cundinamarca ─────────────────────────────────────
# Bounding box aproximado del municipio (WGS84)
SOPO_BBOX = {
    "lat_min": 4.85,
    "lat_max": 4.96,
    "lon_min": -73.99,
    "lon_max": -73.89,
}

SOPO_CENTRO = (4.9075, -73.9384)  # (lat, lon)

# ─── Imágenes Sentinel-2 ─────────────────────────────────────────────────────
# AWS Open Data Registry - sin autenticación
SENTINEL_AWS_BUCKET = "sentinel-s2-l2a"
SENTINEL_AWS_REGION = "us-west-2"
SENTINEL_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]  # 6 bandas HLS-compatible
SENTINEL_RESOLUTION = 10  # metros

# ─── Modelo IA ────────────────────────────────────────────────────────────────
# Modelo público de Hugging Face (no requiere token para modelos públicos)
MODELO_BASE = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
MODELO_NOMBRE = "prithvi_eo_300m"

# Hiperparámetros LoRA
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# ─── Entrenamiento ────────────────────────────────────────────────────────────
EPOCHS = 30
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 2

# ─── Inferencia ───────────────────────────────────────────────────────────────
TILE_SIZE = 224
TILE_OVERLAP = 0.5  # 50% solapamiento
CONFIDENCE_THRESHOLD = 0.5

# ─── Leyenda CORINE Land Cover adaptada para Colombia (IDEAM) ────────────────
CLASES_COBERTURA = {
    0: {"nombre": "Bosque denso", "color": "#006400"},
    1: {"nombre": "Bosque abierto", "color": "#228B22"},
    2: {"nombre": "Bosque fragmentado", "color": "#90EE90"},
    3: {"nombre": "Herbazal", "color": "#FFD700"},
    4: {"nombre": "Pastos", "color": "#ADFF2F"},
    5: {"nombre": "Agricultura", "color": "#FF8C00"},
    6: {"nombre": "Zonas urbanas", "color": "#FF0000"},
    7: {"nombre": "Cuerpos de agua", "color": "#0000FF"},
    8: {"nombre": "Suelo desnudo", "color": "#8B4513"},
    9: {"nombre": "Nubes/sombra", "color": "#808080"},
}

NUM_CLASES = len(CLASES_COBERTURA)

# ─── Normalización HLS (media y std para las 6 bandas) ────────────────────────
# Valores del dataset HLS usado en pre-entrenamiento de Prithvi
HLS_MEAN = [0.033, 0.033, 0.033, 0.033, 0.033, 0.033]
HLS_STD = [0.033, 0.033, 0.033, 0.033, 0.033, 0.033]

# ─── Temporada seca recomendada (menos nubes) ─────────────────────────────────
TEMPORADA_SECA_MESES = [12, 1, 2, 3]  # Diciembre a marzo

# ─── Configuración de GPU ────────────────────────────────────────────────────
USE_AMP = True  # Mixed precision para ahorrar VRAM
DEVICE = "cuda"  # "cuda" o "cpu"
