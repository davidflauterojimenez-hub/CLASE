"""
Inferencia con el modelo entrenado sobre tiles de Sopó.
Genera predicciones de cobertura terrestre por tile.

Uso:
    python 04_inferencia.py
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from config import (
    TILES_DIR, MODELO_DIR, MODELO_BASE, LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
    NUM_CLASES, USE_AMP, DEVICE, CONFIDENCE_THRESHOLD
)
from 03_entrenar import PrithviSegmentationModel, cargar_modelo_prithvi, configurar_lora


def cargar_modelo_entrenado():
    """Carga el modelo entrenado con LoRA."""
    print(f"Cargando modelo entrenado desde: {MODELO_DIR}")

    # Cargar backbone
    backbone = cargar_modelo_prithvi()
    backbone = configurar_lora(backbone, LORA_RANK, LORA_ALPHA, LORA_DROPOUT)

    # Crear modelo completo
    model = PrithviSegmentationModel(backbone, NUM_CLASES)

    # Cargar pesos entrenados
    ruta_pesos = os.path.join(MODELO_DIR, "mejor_modelo.pth")
    if os.path.exists(ruta_pesos):
        model.load_state_dict(torch.load(ruta_pesos, map_location=DEVICE))
        print(f"Pesos cargados desde: {ruta_pesos}")
    else:
        print(f"No se encontró {ruta_pesos}. Usando pesos pre-entrenados.")

    model = model.to(DEVICE)
    model.eval()
    return model


def inferir_tile(model, tile):
    """
    Inferencia sobre un solo tile.
    tile: numpy array (6, 224, 224)
    Retorna: predicción (224, 224) y probabilidades (10, 224, 224)
    """
    tile_tensor = torch.from_numpy(tile).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            output = model(tile_tensor)
            probs = F.softmax(output, dim=1)

    prediccion = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
    probabilidades = probs.squeeze(0).cpu().numpy()

    return prediccion, probabilidades


def inferir_todos_tiles():
    """
    Función principal: ejecuta inferencia sobre todos los tiles.
    """
    print("=" * 60)
    print("INFERENCIA SOBRE TILES DE SOPÓ")
    print("=" * 60)

    # Cargar modelo
    model = cargar_modelo_entrenado()

    # Buscar tiles
    tiles = sorted([f for f in os.listdir(TILES_DIR) if f.endswith(".npy")])
    print(f"Tiles encontrados: {len(tiles)}")

    if not tiles:
        print("No hay tiles. Ejecuta 02_preprocesar.py primero.")
        return

    # Crear directorio de predicciones
    pred_dir = os.path.join(MODELO_DIR, "predicciones")
    os.makedirs(pred_dir, exist_ok=True)

    # Inferencia
    for tile_file in tqdm(tiles, desc="Inferencia"):
        tile_path = os.path.join(TILES_DIR, tile_file)
        tile = np.load(tile_path)

        prediccion, probabilidades = inferir_tile(model, tile)

        # Guardar predicción
        nombre_base = tile_file.replace(".npy", "")
        ruta_pred = os.path.join(pred_dir, f"{nombre_base}_pred.npy")
        ruta_prob = os.path.join(pred_dir, f"{nombre_base}_prob.npy")

        np.save(ruta_pred, prediccion.astype(np.uint8))
        np.save(ruta_prob, probabilidades.astype(np.float32))

    print(f"\nPredicciones guardadas en: {pred_dir}")
    print("=" * 60)


if __name__ == "__main__":
    inferir_todos_tiles()
