# Generación de Imágenes con DCGAN — Proyecto de IA Generativa

Este proyecto implementa una Red Generativa Antagónica (DCGAN) entrenada sobre Fashion-MNIST, un dataset compuesto por 60,000 imágenes de entrenamiento en escala de grises redimensionadas a 64×64 píxeles de distintas prendas de vestir. Incluye un notebook de entrenamiento, tres experimentos comparativos con evaluación cuantitativa y una aplicación interactiva en Streamlit para generar imágenes utilizando los modelos entrenados.

“La problemática abordada es la necesidad de generar rápidamente propuestas visuales de prendas para prototipado en moda digital y comercio electrónico. Esto permite acelerar procesos creativos, reducir costos y apoyar a diseñadores junior.”

---

## Características principales

| Característica | Descripción |
|----------------|-------------|
| **Implementación** | DCGAN completa con PyTorch |
| **Dataset** | Fashion-MNIST redimensionado a 64×64 |
| **Experimentos** | 3 configuraciones comparativas |
| **Evaluación** | Métricas cuantitativas (Realism, Diversity, IS Proxy) |
| **Interfaz** | Aplicación Streamlit para generación interactiva |
| **Modularidad** | Código organizado y fácil de extender |

---

## Estructura del proyecto

```
IAGENERATIVA_DEEPLEARNING/
│
├── App.py                              # Aplicación Streamlit
├── requirements.txt                    # Dependencias
├── IAGENERATIVA_DEEPLEARNING.ipynb     # Notebook de entrenamiento
│
├── modelos/                            # Modelos DCGAN generados
│     ├── exp1_baseline.pth
│     ├── exp2_mas_epocas.pth
│     └── exp3_lrD_bajo.pth
│
├── imagenes_exp1_baseline/             # Imágenes por época
├── imagenes_exp2_mas_epocas/
├── imagenes_exp3_lrD_bajo/
│
└── README.md
```

---

## Arquitectura de la DCGAN

### Parámetros generales

| Parámetro | Valor |
|-----------|-------|
| **Tamaño de imagen** | 64×64 |
| **Vector latente (nz)** | 100 |
| **Canales (nc)** | 1 (escala de grises) |
| **Filtros base Generador (ngf)** | 32 |
| **Filtros base Discriminador (ndf)** | 32 |
| **Batch size** | 128 |
| **Optimizador** | Adam (beta1=0.5, beta2=0.999) |

### Generador

| Componente | Descripción |
|------------|-------------|
| **Entrada** | Vector de ruido `z` (100×1×1) |
| **Capas** | 5 capas ConvTranspose2d con BatchNorm y LeakyReLU (0.2) |
| **Progresión** | 4×4 → 8×8 → 16×16 → 32×32 → 64×64 |
| **Salida** | Imagen 64×64×1, normalizada con `Tanh` [-1, 1] |

### Discriminador

| Componente | Descripción |
|------------|-------------|
| **Entrada** | Imagen 64×64×1 |
| **Capas** | 5 capas Conv2d con BatchNorm y LeakyReLU (0.2) |
| **Progresión** | 64×64 → 32×32 → 16×16 → 8×8 → 4×4 → 1×1 |
| **Salida** | Probabilidad real/falso con Sigmoid [0, 1] |

---

## Entrenamiento

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Luego abre y ejecuta:

```bash
jupyter notebook IAGENERATIVA_DEEPLEARNING.ipynb
```

El notebook guarda los modelos automáticamente en la carpeta `modelos/`:

| Modelo | Descripción |
|--------|-------------|
| `exp1_baseline.pth` | Experimento 1: Configuración baseline |
| `exp2_mas_epocas.pth` | Experimento 2: Mayor número de épocas |
| `exp3_lrD_bajo.pth` | Experimento 3: Learning rate del discriminador reducido |

---

## Resultados de los experimentos

### Dataset: Fashion-MNIST

| Propiedad | Valor |
|-----------|-------|
| **Dimensiones originales** | 28×28 píxeles |
| **Dimensiones procesadas** | 64×64 píxeles (redimensionado) |
| **Formato** | Escala de grises (1 canal) |
| **Normalización** | [-1, 1] mediante `Normalize((0.5,), (0.5,))` |
| **Clases** | 10 (camisetas, pantalones, suéteres, vestidos, abrigos, sandalias, camisas, zapatillas, bolsos, botas) |
| **Total imágenes** | 60,000 (entrenamiento) |

### Experimento 1 – Baseline

| Parámetro | Valor |
|-----------|-------|
| **Épocas** | 10 |
| **Learning Rate (G)** | 0.0002 |
| **Learning Rate (D)** | 0.0002 |
| **Max batches por época** | 200 |

**Resultados:** Entrenamiento estándar con formas reconocibles y presencia de ruido.

### Experimento 2 – Más épocas

| Parámetro | Valor |
|-----------|-------|
| **Épocas** | 20 |
| **Learning Rate (G)** | 0.0002 |
| **Learning Rate (D)** | 0.0002 |
| **Max batches por época** | 200 |

**Resultados:** Mejor definición de detalles y reducción significativa de artefactos.

### Experimento 3 – Learning rate del discriminador reducido

| Parámetro | Valor |
|-----------|-------|
| **Épocas** | 20 |
| **Learning Rate (G)** | 0.0002 |
| **Learning Rate (D)** | 0.0001 |
| **Max batches por época** | 200 |

**Resultados:** Mayor estabilidad en el entrenamiento y formas más consistentes.

---

## Métricas de evaluación

El proyecto incluye un clasificador CNN para evaluar cuantitativamente la calidad de las imágenes generadas:

| Métrica | Descripción |
|---------|-------------|
| **Realism Score** | Confianza promedio del clasificador en las imágenes generadas |
| **Diversity Score** | Proporción de clases únicas predichas sobre el total de clases (10) |
| **IS Proxy** | Aproximación al Inception Score basada en la distribución de predicciones |

---

## Uso de la aplicación Streamlit

Ejecutar:

```bash
streamlit run App.py
```

### Funcionalidades

| Función | Descripción |
|---------|-------------|
| **Selección de modelo** | Elegir entre exp1_baseline, exp2_mas_epocas, exp3_lrD_bajo |
| **Cantidad de imágenes** | Definir cuántas imágenes generar |
| **Visualización** | Ver las muestras en cuadrícula |

---

## Instalación completa

```bash
git clone https://github.com/tuusuario/IAGENERATIVA_DEEPLEARNING.git
cd IAGENERATIVA_DEEPLEARNING
pip install -r requirements.txt
streamlit run App.py
```

---

## Dependencias principales

| Librería | Versión mínima | Propósito |
|----------|----------------|-----------|
| **Python** | 3.8+ | Lenguaje base |
| **PyTorch** | 1.12+ | Framework de deep learning |
| **Torchvision** | 0.13+ | Datasets y transformaciones |
| **Streamlit** | 1.20+ | Interfaz web |
| **Matplotlib** | 3.5+ | Visualización |
| **NumPy** | 1.21+ | Operaciones numéricas |
| **tqdm** | 4.64+ | Barras de progreso |
| **Pandas** | 1.4+ | Análisis de resultados |

---

## Licencia

Este proyecto se distribuye bajo la licencia MIT.

