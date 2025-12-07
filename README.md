Generación de Imágenes con DCGAN — Proyecto de IA Generativa

Este proyecto implementa una Red Generativa Antagónica (DCGAN) entrenada sobre Fashion-MNIST, un dataset compuesto por 70,000 imágenes en escala de grises (28×28) de distintas prendas de vestir.
Incluye un notebook de entrenamiento, tres experimentos comparativos y una aplicación interactiva en Streamlit para generar imágenes utilizando los modelos entrenados.

Características principales

Implementación completa de una DCGAN con PyTorch

Entrenamiento sobre Fashion-MNIST

Tres experimentos comparativos:

Experimento 1 – Baseline

Experimento 2 – Más épocas

Experimento 3 – Tasa de aprendizaje del discriminador reducida

Aplicación en Streamlit para usar los generadores (.pth)

Código modular y fácil de extender

Estructura del proyecto
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

Arquitectura de la DCGAN
Generador

Entrada: vector de ruido z (100 dimensiones)

Capas: ConvTranspose2d, BatchNorm, ReLU

Salida: imagen 28×28×1, normalizada con Tanh

Discriminador

Entrada: imagen 28×28×1

Capas: Conv2d, BatchNorm, LeakyReLU

Salida: probabilidad real/falso (sigmoid)

Entrenamiento

Para ejecutar el notebook:

pip install torch torchvision matplotlib


Luego abre y ejecuta:

IAGENERATIVA_DEEPLEARNING.ipynb


El notebook guarda los modelos automáticamente en la carpeta modelos/:

exp1_baseline.pth

exp2_mas_epocas.pth

exp3_lrD_bajo.pth

Resultados de los experimentos
Dataset: Fashion-MNIST

Imágenes: 28×28 píxeles

Escala de grises

Clases: camisetas, zapatos, bolsos, abrigos, sneakers, etc.

Experimento 1 – Baseline

Entrenamiento estándar: formas reconocibles, con algo de ruido.
(Agregar aquí la imagen exp1_result.png)

Experimento 2 – Más épocas

Mejor definición y reducción de artefactos.
(Agregar aquí exp2_result.png)

Experimento 3 – lrD más bajo

Mejor estabilidad del discriminador y formas más consistentes.
(Agregar aquí exp3_result.png)

Uso de la aplicación Streamlit

Ejecutar:

streamlit run App.py


La aplicación permite:

Seleccionar uno de los tres modelos

Elegir cuántas imágenes generar

Visualizar las muestras en cuadrícula

Instalación completa
git clone https://github.com/tuusuario/IAGENERATIVA_DEEPLEARNING.git
cd IAGENERATIVA_DEEPLEARNING
pip install -r requirements.txt
streamlit run App.py

Dependencias principales

Python 3.9+

PyTorch

Torchvision

Streamlit

Matplotlib

NumPy

Contribuciones

Las contribuciones son bienvenidas mediante Issues o Pull Requests.

Licencia

Este proyecto se distribuye bajo la licencia MIT.
