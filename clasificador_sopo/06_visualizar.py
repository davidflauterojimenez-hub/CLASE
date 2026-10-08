"""
Visualización interactiva de resultados.
Genera un mapa interactivo con Folium y una comparación con imagen RGB.

Uso:
    python 06_visualizar.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from config import (
    SOPO_CENTRO, RESULTADOS_DIR, MODELO_DIR, CLASES_COBERTURA, DATOS_DIR
)


def crear_mapa_interactivo():
    """
    Crea un mapa interactivo con Folium mostrando la clasificación.
    """
    print("=" * 60)
    print("GENERANDO MAPA INTERACTIVO")
    print("=" * 60)

    try:
        import folium
        from folium import plugins
    except ImportError:
        print("Folium no instalado. Instala con: pip install folium")
        return None

    # Cargar mosaico de predicciones
    ruta_mosaico = os.path.join(RESULTADOS_DIR, "mosaico_predicciones.npy")
    if not os.path.exists(ruta_mosaico):
        print("No se encontró el mosaico. Ejecuta 05_postprocesar.py primero.")
        return None

    mosaico = np.load(ruta_mosaico)

    # Crear mapa centrado en Sopó
    m = folium.Map(
        location=SOPO_CENTRO,
        zoom_start=13,
        tiles="OpenStreetMap"
    )

    # Añadir capa de imagen satelital
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Imagen satelital",
        overlay=False,
        control=True
    ).add_to(m)

    # Crear imagen overlay del mosaico
    # Convertir mosaico a imagen RGBA
    rgba = np.zeros((mosaico.shape[0], mosaico.shape[1], 4), dtype=np.uint8)

    for clase_id, info in CLASES_COBERTURA.items():
        mask = mosaico == clase_id
        hex_color = info["color"].lstrip("#")
        rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        rgba[mask, :3] = rgb
        rgba[mask, 3] = 180  # Transparencia

    # Guardar imagen temporal
    from PIL import Image
    img_path = os.path.join(RESULTADOS_DIR, "mosaico_rgba.png")
    Image.fromarray(rgba).save(img_path)

    # Añadir overlay al mapa
    bounds = [[4.85, -73.99], [4.96, -73.89]]  # lat_min/lon_min, lat_max/lon_max
    folium.raster_layers.ImageOverlay(
        image=img_path,
        bounds=bounds,
        opacity=0.7,
        name="Cobertura terrestre",
        interactive=True
    ).add_to(m)

    # Añadir leyenda
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border: 2px solid grey;
                border-radius: 5px; font-size: 12px;">
    <b>Cobertura Terrestre - Sopó</b><br>
    """
    for clase_id, info in CLASES_COBERTURA.items():
        legend_html += f'<i style="background:{info["color"]};width:12px;height:12px;display:inline-block;"></i> {info["nombre"]}<br>'
    legend_html += "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))

    # Control de capas
    folium.LayerControl().add_to(m)

    # Guardar mapa
    ruta_mapa = os.path.join(RESULTADOS_DIR, "mapa_interactivo.html")
    m.save(ruta_mapa)
    print(f"Mapa interactivo guardado: {ruta_mapa}")

    return ruta_mapa


def crear_comparacion_rgb():
    """
    Crea una comparación visual entre imagen RGB y clasificación.
    """
    print("\nGenerando comparación RGB vs Clasificación...")

    # Buscar imagen RGB de los datos
    rgb_files = sorted([f for f in os.listdir(DATOS_DIR) if f.endswith(".npy") and "composicion" in f])

    if not rgb_files:
        print("No se encontró composición temporal RGB")
        return None

    # Cargar composición
    rgb_path = os.path.join(DATOS_DIR, rgb_files[0])
    rgb_data = np.load(rgb_path)

    # Cargar mosaico
    ruta_mosaico = os.path.join(RESULTADOS_DIR, "mosaico_predicciones.npy")
    if not os.path.exists(ruta_mosaico):
        print("No se encontró el mosaico de predicciones")
        return None

    mosaico = np.load(ruta_mosaico)

    # Crear figura
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Imagen RGB (usar primeras 3 bandas normalizadas)
    rgb_display = rgb_data[:3].transpose(1, 2, 0)
    rgb_display = np.clip(rgb_display, 0, 1)
    axes[0].imshow(rgb_display)
    axes[0].set_title("Imagen Sentinel-2 (RGB)", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Clasificación
    # Crear imagen de colores
    class_img = np.zeros((mosaico.shape[0], mosaico.shape[1], 3), dtype=np.uint8)
    for clase_id, info in CLASES_COBERTURA.items():
        mask = mosaico == clase_id
        hex_color = info["color"].lstrip("#")
        rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        class_img[mask] = rgb

    axes[1].imshow(class_img)
    axes[1].set_title("Clasificación de Cobertura Terrestre", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # Leyenda
    patches = [
        mpatches.Patch(color=info["color"], label=info["nombre"])
        for info in CLASES_COBERTURA.values()
    ]
    axes[1].legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)

    plt.tight_layout()

    ruta_comparacion = os.path.join(RESULTADOS_DIR, "comparacion_rgb_clasificacion.png")
    plt.savefig(ruta_comparacion, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Comparación guardada: {ruta_comparacion}")
    return ruta_comparacion


def visualizar():
    """
    Función principal de visualización.
    """
    # Mapa interactivo
    crear_mapa_interactivo()

    # Comparación RGB
    crear_comparacion_rgb()

    print(f"\n{'=' * 60}")
    print("Visualización completada")
    print(f"Resultados en: {RESULTADOS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    visualizar()
