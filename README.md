🧬 Generación de Imágenes con DCGAN — Proyecto de IA Generativa

Este proyecto implementa una Red Generativa Antagónica (GAN) del tipo DCGAN entrenada para generar imágenes a partir del dataset Fashion-MNIST, que contiene 70,000 imágenes en escala de grises de prendas de ropa (10 clases).
Incluye:

Notebook de entrenamiento con 3 experimentos comparativos

Aplicación interactiva en Streamlit para usar los generadores entrenados

Modelos .pth listos para probar

Código modular, limpio y fácil de extender

📌 Características principales
✔ Entrenamiento completo de DCGAN

Implementación de Generador y Discriminador basados en convoluciones transpuestas.

Normalización por batch, pesos inicializados tipo DCGAN y arquitectura recomendada por el paper original.

Registro continuo del entrenamiento con torch.utils.make_grid.

✔ Tres experimentos de entrenamiento

Experimento 1 – Baseline: configuración clásica de DCGAN

Experimento 2 – Más épocas: se entrena por más tiempo para evaluar mejora

Experimento 3 – lrD más bajo: se ajusta la tasa de aprendizaje del discriminador

✔ Aplicación Streamlit integrada

Genera imágenes con cualquier modelo entrenado

Slider para generar múltiples imágenes

Visualización en cuadrícula

Soporte para GPU si está disponible

🚀 Demo (Streamlit)

Ejecuta la App:

streamlit run App.py

📁 Estructura del proyecto
IAGENERATIVA_DEEPLEARNING/
│
├── App.py                         # Aplicación Streamlit
├── requirements.txt               # Dependencias
├── IAGENERATIVA_DEEPLEARNING.ipynb  # Notebook de entrenamiento
│
├── modelos/                       # Modelos DCGAN guardados
│     ├── exp1_baseline.pth
│     ├── exp2_mas_epocas.pth
│     └── exp3_lrD_bajo.pth
│
└── README.md

🏗 Arquitectura de la DCGAN
Generador

Entrada: vector ruido z (100 dimensiones)

Capas: ConvTranspose2d + BatchNorm + ReLU

Salida: imagen 28×28 en escala de grises con Tanh (propio de Fashion-MNIST)

Discriminador

Entrada: imagen real/falsa 28×28×1

Capas: Conv2d + BatchNorm + LeakyReLU

Salida: probabilidad real/falso

🔥 Entrenamiento

Desde el notebook:

!pip install torch torchvision matplotlib


Luego ejecuta todas las celdas del archivo:

IAGENERATIVA_DEEPLEARNING.ipynb


Los modelos se guardan automáticamente en:

/modelos


con los nombres:

exp1_baseline.pth

exp2_mas_epocas.pth

exp3_lrD_bajo.pth

🧪 Resultados de los experimentos
Dataset: Fashion-MNIST

Imágenes 28×28

Escala de grises

Clases como: camiseta, zapato, abrigo, bolso, sneaker, etc.

Experimento 1 – Baseline

20 épocas

Buen comienzo; formas reconocibles pero con ruido

📸 Placeholder
(Agrega aquí exp1_result.png)

Experimento 2 – Más épocas

40 épocas

Imágenes más nítidas

Mejor definición de contornos y formas

📸 Placeholder
(Agrega aquí exp2_result.png)

Experimento 3 – lrD más bajo

Discriminador más estable

Menos artefactos y mayor coherencia visual

📸 Placeholder
(Agrega aquí exp3_result.png)

🎛 Uso de la App
Seleccionar el modelo

En el panel lateral de Streamlit puedes elegir entre:

Experimento 1 – Baseline

Experimento 2 – Más épocas

Experimento 3 – lrD más bajo

Generar imágenes

Selecciona el modelo

Ajusta cuántas imágenes generar

Haz clic en “Generar imágenes”

🛠 Instalación y ejecución
1. Clonar el repositorio
git clone https://github.com/tuusuario/IAGENERATIVA_DEEPLEARNING.git
cd IAGENERATIVA_DEEPLEARNING

2. Instalar dependencias
pip install -r requirements.txt

3. Ejecutar la App
streamlit run App.py

📦 Dependencias principales

PyTorch

Torchvision

Streamlit

Matplotlib

NumPy


📜 Licencia

Este proyecto se distribuye bajo la licencia MIT.
