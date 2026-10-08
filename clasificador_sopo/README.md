# Clasificador de Cobertura Terrestre - Sopó, Cundinamarca

Programa de clasificación de cobertura terrestre usando inteligencia artificial geoespacial de código abierto. **Sin autenticación requerida.**

## Características

- **Modelo**: Prithvi-EO-2.0-300M (NASA/IBM) con fine-tuning LoRA
- **Datos**: Sentinel-2 L2A desde AWS Open Data (público, sin login)
- **Resolución**: 10 metros por píxel
- **Leyenda**: 10 clases CORINE Land Cover adaptada (IDEAM)
- **GPU**: Optimizado para 8GB VRAM

## Requisitos

- Python 3.10+
- GPU con 8GB+ VRAM (o CPU, más lento)
- ~10GB espacio en disco

## Instalación

```bash
# Crear entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso rápido

```bash
# Pipeline completo
python main.py

# Ejecutar paso específico
python main.py --paso 1  # Descargar imágenes
python main.py --paso 2  # Preprocesar
python main.py --paso 3  # Entrenar modelo
python main.py --paso 4  # Inferencia
python main.py --paso 5  # Postprocesar
python main.py --paso 6  # Visualizar
```

## Estructura del proyecto

```
clasificador_sopo/
├── main.py                  # Pipeline principal
├── config.py                # Configuración
├── 01_descargar_imagenes.py # Descarga Sentinel-2
├── 02_preprocesar.py        # Tiles y normalización
├── 03_entrenar.py           # Fine-tuning LoRA
├── 04_inferencia.py         # Predicción
├── 05_postprocesar.py       # Mosaico y estadísticas
├── 06_visualizar.py         # Mapa interactivo
├── requirements.txt         # Dependencias
├── BITACORA.md              # Registro de decisiones
├── datos/                   # Imágenes descargadas
├── tiles/                   # Tiles 224x224
├── modelo/                  # Pesos del modelo
└── resultados/              # Salidas finales
```

## Salidas

| Archivo | Descripción |
|---------|-------------|
| `resultados/cobertura_sopo.tif` | GeoTIFF clasificado |
| `resultados/cobertura_sopo.shp` | Shapefile con polígonos |
| `resultados/estadisticas_cobertura.json` | Estadísticas por clase |
| `resultados/grafico_coberturas.png` | Gráfico de barras |
| `resultados/mapa_interactivo.html` | Mapa web interactivo |
| `resultados/comparacion_rgb_clasificacion.png` | Comparación visual |

## Leyenda de clases

| Código | Cobertura | Color |
|--------|-----------|-------|
| 0 | Bosque denso | Verde oscuro |
| 1 | Bosque abierto | Verde medio |
| 2 | Bosque fragmentado | Verde claro |
| 3 | Herbazal | Amarillo |
| 4 | Pastos | Verde amarillento |
| 5 | Agricultura | Naranja |
| 6 | Zonas urbanas | Rojo |
| 7 | Cuerpos de agua | Azul |
| 8 | Suelo desnudo | Marrón |
| 9 | Nubes/sombra | Gris |

## Cómo funciona

1. **Descarga**: Obtiene imágenes Sentinel-2 L2A desde AWS Open Data (S3 público)
2. **Preprocesamiento**: Crea tiles 224×224 con normalización HLS
3. **Entrenamiento**: Ajusta el modelo Prithvi con LoRA (rank=8)
4. **Inferencia**: Clasifica cada tile en 10 categorías
5. **Postprocesamiento**: Genera mosaico, estadísticas y exportaciones
6. **Visualización**: Crea mapa interactivo y gráficos

## Notas

- **Sin tokens**: Los datos de Sentinel-2 y el modelo Prithvi son públicos
- **Temporada seca**: Se recomienda enero-marzo para menos nubes
- **Mejorar precisión**: Hacer fine-tuning con datos etiquetados de IDEAM o MapBiomas
- **Tiempo estimado**: 30-60 minutos en GPU 8GB

## Referencias

- [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0)
- [AWS Sentinel-2 Open Data](https://registry.opendata.aws/sentinel-2/)
- [TerraTorch](https://github.com/IBM/terratorch)
- [MapBiomas Colombia](https://colombia.mapbiomas.org)
