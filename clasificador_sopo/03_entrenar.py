"""
Entrenamiento del modelo Prithvi-EO-2.0 con fine-tuning LoRA.
Optimizado para GPU 8GB con mixed precision.

Uso:
    python 03_entrenar.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import (
    TILES_DIR, MODELO_DIR, MODELO_BASE, LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
    EPOCHS, BATCH_SIZE, GRADIENT_ACCUMULATION, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, NUM_WORKERS, NUM_CLASES, USE_AMP, DEVICE
)


class TilesDataset(Dataset):
    """Dataset de tiles para entrenamiento."""

    def __init__(self, tiles_dir, labels_dir=None):
        self.tiles_dir = tiles_dir
        self.tiles = sorted([f for f in os.listdir(tiles_dir) if f.endswith(".npy")])
        self.labels_dir = labels_dir

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = np.load(os.path.join(self.tiles_dir, self.tiles[idx]))
        tile = torch.from_numpy(tile).float()

        # Si no hay etiquetas, usar dummy (modo inferencia)
        if self.labels_dir and os.path.exists(os.path.join(self.labels_dir, self.tiles[idx])):
            label = np.load(os.path.join(self.labels_dir, self.tiles[idx]))
            label = torch.from_numpy(label).long()
        else:
            label = torch.zeros(tile.shape[1], tile.shape[2], dtype=torch.long)

        return tile, label


def cargar_modelo_prithvi():
    """
    Carga el modelo Prithvi-EO-2.0-300M desde Hugging Face.
    No requiere token para modelos públicos.
    """
    print(f"Cargando modelo: {MODELO_BASE}")

    try:
        from terratorch.registry import BACKBONE_REGISTRY

        backbone = BACKBONE_REGISTRY.build(MODELO_BASE)
        print("Modelo cargado desde TerraTorch")
        return backbone
    except Exception as e:
        print(f"Error cargando desde TerraTorch: {e}")
        print("Intentando carga alternativa...")

        # Carga alternativa con transformers
        from transformers import AutoModel

        model = AutoModel.from_pretrained(MODELO_BASE, trust_remote_code=True)
        return model


def configurar_lora(model, rank=8, alpha=16, dropout=0.1):
    """
    Configura LoRA (Low-Rank Adaptation) para fine-tuning eficiente.
    Reduce los parámetros entrenables en ~70%.
    """
    try:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["qkv", "proj"],
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )

        model = get_peft_model(model, lora_config)
        print(f"LoRA configurado: rank={rank}, alpha={alpha}, dropout={dropout}")

        # Mostrar parámetros entrenables
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parámetros totales: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

        return model
    except ImportError:
        print("PEFT no disponible. Fine-tuning completo (más lento y memoria).")
        return model


class PrithviSegmentationModel(nn.Module):
    """Modelo de segmentación con backbone Prithvi y decoder UPerNet."""

    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        # Decoder simple (FCN) para segmentación
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, x):
        # x: (B, 6, H, W)
        features = self.backbone(x)

        # Extraer feature map
        if isinstance(features, dict):
            x = features.get("features", list(features.values())[0])
        elif isinstance(features, (list, tuple)):
            x = features[-1]

        # Redimensionar si es necesario
        if x.dim() == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)

        # Decoder
        out = self.decoder(x)

        # Upsample a tamaño original
        out = nn.functional.interpolate(
            out, size=(224, 224), mode="bilinear", align_corners=False
        )
        return out


def entrenar_epoca(model, dataloader, optimizer, scaler, device, epoch):
    """Entrena una época."""
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(dataloader, desc=f"Época {epoch}")
    optimizer.zero_grad()

    for i, (tiles, labels) in enumerate(pbar):
        tiles = tiles.to(device)
        labels = labels.to(device)

        with autocast(enabled=USE_AMP):
            outputs = model(tiles)
            loss = criterion(outputs, labels)
            loss = loss / GRADIENT_ACCUMULATION

        scaler.scale(loss).backward()

        if (i + 1) % GRADIENT_ACCUMULATION == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * GRADIENT_ACCUMULATION
        pbar.set_postfix({"loss": loss.item() * GRADIENT_ACCUMULATION})

    return total_loss / len(dataloader)


def validar(model, dataloader, device):
    """Valida el modelo."""
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for tiles, labels in tqdm(dataloader, desc="Validación"):
            tiles = tiles.to(device)
            labels = labels.to(device)

            with autocast(enabled=USE_AMP):
                outputs = model(tiles)
                loss = criterion(outputs, labels)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def entrenar():
    """
    Función principal de entrenamiento.
    """
    print("=" * 60)
    print("ENTRENAMIENTO DEL MODELO PRITHVI-EO-2.0 CON LORA")
    print("=" * 60)
    print(f"Dispositivo: {DEVICE}")
    print(f"Modelo base: {MODELO_BASE}")
    print(f"LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
    print(f"Épocas: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Mixed precision: {USE_AMP}")
    print("-" * 60)

    # Verificar GPU
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("GPU no disponible, usando CPU")
        DEVICE = "cpu"

    # Cargar datos
    dataset = TilesDataset(TILES_DIR)
    print(f"Tiles encontrados: {len(dataset)}")

    if len(dataset) == 0:
        print("No hay tiles. Ejecuta 02_preprocesar.py primero.")
        return

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Cargar modelo
    backbone = cargar_modelo_prithvi()
    backbone = configurar_lora(backbone, LORA_RANK, LORA_ALPHA, LORA_DROPOUT)

    model = PrithviSegmentationModel(backbone, NUM_CLASES)
    model = model.to(DEVICE)

    # Optimizador
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scaler = GradScaler(enabled=USE_AMP)

    # Entrenamiento
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Época {epoch}/{EPOCHS} ---")

        train_loss = entrenar_epoca(model, train_loader, optimizer, scaler, DEVICE, epoch)
        val_loss = validar(model, val_loader, DEVICE)

        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ruta_modelo = os.path.join(MODELO_DIR, "mejor_modelo.pth")
            torch.save(model.state_dict(), ruta_modelo)
            print(f"Mejor modelo guardado en: {ruta_modelo}")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping activado")
            break

    print(f"\n{'=' * 60}")
    print("Entrenamiento completado")
    print(f"Mejor modelo guardado en: {MODELO_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    entrenar()
