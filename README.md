# Generación de Imágenes con DCGAN — Proyecto de IA Generativa

Este proyecto implementa una Red Generativa Antagónica (DCGAN) entrenada sobre Fashion-MNIST, un dataset compuesto por 70,000 imágenes en escala de grises (28×28) de distintas prendas de vestir. Incluye un notebook de entrenamiento, tres experimentos comparativos y una aplicación interactiva en Streamlit para generar imágenes utilizando los modelos entrenados.

---

## Características principales

| Característica | Descripción |
|----------------|-------------|
| **Implementación** | DCGAN completa con PyTorch |
| **Dataset** | Fashion-MNIST (70,000 imágenes) |
| **Experimentos** | 3 configuraciones comparativas |
| **Interfaz** | Aplicación Streamlit para generación interactiva |
| **Modularidad** | Código organizado y fácil de extender |

---

## Estructura del proyecto

```
IAGENERATIVA_DEEPLEARNING/
│
├── App.py                         # Aplicación Streamlit
├── requirements.txt               # Dependencias
├── IAGENERATIVA_DEEPLEARNING.ipynb  # Notebook de entrenamiento
│
├── modelos/                       # Modelos DCGAN generados
│     ├── exp1_baseline.pth
│     ├── exp2_mas_epocas.pth
│     └── exp3_lrD_bajo.pth
│
└── README.md
```

---

## Arquitectura de la DCGAN

### Generador

| Componente | Descripción |
|------------|-------------|
| **Entrada** | Vector de ruido `z` (100 dimensiones) |
| **Capas** | ConvTranspose2d, BatchNorm, ReLU |
| **Salida** | Imagen 28×28×1, normalizada con `Tanh` |

### Discriminador

| Componente | Descripción |
|------------|-------------|
| **Entrada** | Imagen 28×28×1 |
| **Capas** | Conv2d, BatchNorm, LeakyReLU |
| **Salida** | Probabilidad real/falso (sigmoid) |

---

## Entrenamiento

Para ejecutar el notebook:

```bash
pip install torch torchvision matplotlib
```

Luego abre y ejecuta:

```bash
IAGENERATIVA_DEEPLEARNING.ipynb
```

El notebook guarda los modelos automáticamente en la carpeta `modelos/`:

| Modelo | Descripción |
|--------|-------------|
| `exp1_baseline.pth` | Experimento 1: Configuración baseline |
| `exp2_mas_epocas.pth` | Experimento 2: Mayor número de épocas |
| `exp3_lrD_bajo.pth` | Experimento 3: Learning rate reducido |

---

## Resultados de los experimentos

### Dataset: Fashion-MNIST

| Propiedad | Valor |
|-----------|-------|
| **Dimensiones** | 28×28 píxeles |
| **Formato** | Escala de grises |
| **Clases** | Camisetas, zapatos, bolsos, abrigos, sneakers, etc. |
| **Total imágenes** | 70,000 |

### Experimento 1 – Baseline

| Parámetro | Valor |
|-----------|-------|
| **Épocas** | 50 |
| **Learning Rate (G)** | 0.0002 |
| **Learning Rate (D)** | 0.0002 |

**Resultados:** Entrenamiento estándar con formas reconocibles y algo de ruido.

### Experimento 2 – Más épocas

| Parámetro | Valor |
|-----------|-------|
| **Épocas** | 100 |
| **Learning Rate (G)** | 0.0002 |
| **Learning Rate (D)** | 0.0002 |

**Resultados:** Mejor definición y reducción de artefactos.

### Experimento 3 – lrD más bajo

| Parámetro | Valor |
|-----------|-------|
| **Épocas** | 50 |
| **Learning Rate (G)** | 0.0002 |
| **Learning Rate (D)** | 0.0001 |

**Resultados:** Mejor estabilidad del discriminador y formas más consistentes.

---

## Uso de la aplicación Streamlit

Ejecutar:

```bash
streamlit run App.py
```

### Funcionalidades

| Función | Descripción |
|---------|-------------|
| **Selección de modelo** | Elegir entre los tres experimentos |
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
| **Python** | 3.9+ | Lenguaje base |
| **PyTorch** | Latest | Framework de deep learning |
| **Torchvision** | Latest | Datasets y transformaciones |
| **Streamlit** | Latest | Interfaz web |
| **Matplotlib** | Latest | Visualización |
| **NumPy** | Latest | Operaciones numéricas |

---

## Contribuciones

Las contribuciones son bienvenidas mediante Issues o Pull Requests.

---

## Licencia

Este proyecto se distribuye bajo la licencia MIT.
