import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import streamlit as st
from PIL import Image

# ============================
# Configuración básica
# ============================

st.set_page_config(
    page_title="DCGAN Fashion – IA Generativa",
    page_icon="🧥",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* Quitar menú y pie de página de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible;}
    </style>
    """,
    unsafe_allow_html=True
)

# Directorio donde están los modelos dentro del repo
MODELS_DIR = os.path.join(os.path.dirname(__file__), "modelos")

# Hiperparámetros (ajusta si en tu notebook usaste otros)
nz = 100   # tamaño del vector de ruido
ngf = 64   # tamaño base de filtros del generador
nc = 1     # canales de salida (1 = escala de grises)


# ============================
# Definición del Generador
# (igual al usado en el notebook)
# ============================

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: Z (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # ngf x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 3, bias=False),
            nn.Tanh()
            # nc x 28 x 28 (ajustado a Fashion-MNIST)
        )

    def forward(self, x):
        return self.main(x)


# ============================
# Funciones auxiliares
# ============================

@st.cache_resource(show_spinner=False)
def cargar_modelo(nombre_archivo: str):
    """
    Carga un modelo de generador desde modelos/*.pth
    """
    ruta = os.path.join(MODELS_DIR, nombre_archivo)
    if not os.path.exists(ruta):
        return None, f"No se encontró el archivo de modelo: {ruta}"

    device = torch.device("cpu")
    model = Generator(nz=nz, ngf=ngf, nc=nc).to(device)
    state = torch.load(ruta, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, None


def generar_imagenes(modelo: nn.Module, num_imágenes: int, seed: int):
    """
    Genera imágenes sintéticas con el generador entrenado.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cpu")
    noise = torch.randn(num_imágenes, nz, 1, 1, device=device)
    with torch.no_grad():
        fake = modelo(noise).detach().cpu()

    # normalizar a [0,1]
    fake = (fake + 1) / 2

    grid = make_grid(fake, nrow=int(np.sqrt(num_imágenes)), padding=2)
    ndarr = grid.mul(255).clamp(0, 255).byte().numpy()

    # Para 1 canal, está en [C,H,W] con C=1
    if ndarr.shape[0] == 1:
        ndarr = ndarr[0]  # [H,W]
        img = Image.fromarray(ndarr, mode="L")
    else:
        ndarr = np.transpose(ndarr, (1, 2, 0))
        img = Image.fromarray(ndarr)

    return img


# ============================
# Layout de la aplicación
# ============================

st.title("Generación de ropa con DCGAN – Deep Learning Avanzado")
st.caption("Proyecto de IA generativa con Fashion-MNIST | IUDigital")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        Esta aplicación permite **generar prendas de ropa sintéticas** a partir de un modelo
        **DCGAN** entrenado sobre el dataset Fashion-MNIST.

        Puedes seleccionar distintos **experimentos** (baseline, más épocas, tasa de aprendizaje
        del discriminador más baja) y comparar el estilo de las imágenes generadas.
        """
    )

with col2:
    st.info(
        """
        **Autores**  
        - Juliana María Peña Suárez  
        - Juan Esteban Atehortúa Sánchez  
        - Nikol Tamayo Rua  

        **Curso:** Deep Learning Avanzado – IUDigital  
        **Docente:** Laura Alejandra Sánchez
        """,
        icon="👩‍💻",
    )

st.write("---")

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Controles de generación")

experimentos = {
    "Experimento 1 – Baseline": "exp1_baseline.pth",
    "Experimento 2 – Más épocas": "exp2_mas_epocas.pth",
    "Experimento 3 – lrD más bajo": "exp3_lrD_bajo.pth",
}

exp_nombre = st.sidebar.selectbox(
    "Modelo a utilizar",
    list(experimentos.keys()),
)

num_imgs = st.sidebar.slider(
    "Número de imágenes a generar",
    min_value=4,
    max_value=64,
    value=16,
    step=4,
)

seed = st.sidebar.number_input(
    "Semilla aleatoria",
    min_value=0,
    max_value=9999,
    value=42,
    step=1,
)

auto_generar = st.sidebar.checkbox(
    "Generar automáticamente al cambiar parámetros",
    value=True,
)

btn_generar = st.sidebar.button("🎨 Generar imágenes")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Descripción de los experimentos**

    - *Exp 1:* DCGAN baseline (10 épocas).  
    - *Exp 2:* Más épocas de entrenamiento (20 épocas).  
    - *Exp 3:* Misma arquitectura, pero tasa de aprendizaje del discriminador más baja.
    """
)

# ============================
# Zona principal de resultados
# ============================

st.subheader(f"Imágenes generadas – {exp_nombre}")

modelo, error = cargar_modelo(experimentos[exp_nombre])

if error:
    st.error(
        error
        + "\n\nSube los archivos .pth a la carpeta `modelos/` en GitHub para habilitar la generación.",
        icon="⚠️",
    )
    st.stop()

# Lógica de generación
debe_generar = auto_generar or btn_generar

if debe_generar:
    with st.spinner("Generando imágenes..."):
        img = generar_imagenes(modelo, num_imgs, seed)

    st.image(img, caption=f"{num_imgs} prendas sintéticas generadas", use_column_width=True)

    with st.expander("🧪 Detalles del experimento seleccionado"):
        if "Baseline" in exp_nombre:
            st.markdown(
                """
                **Experimento 1 – Baseline**

                - Arquitectura DCGAN estándar.  
                - 10 épocas de entrenamiento.  
                - Buen equilibrio entre realismo y estabilidad, aunque algunas prendas
                  son borrosas o poco definidas.
                """
            )
        elif "Más épocas" in exp_nombre:
            st.markdown(
                """
                **Experimento 2 – Más épocas**

                - Mismo modelo, pero entrenado durante 20 épocas.  
                - Mejora la nitidez y la forma de las prendas.  
                - Ligero riesgo de sobreajuste, pero mantiene buena diversidad.
                """
            )
        else:
            st.markdown(
                """
                **Experimento 3 – lrD más bajo**

                - Se reduce la tasa de aprendizaje del discriminador.  
                - Permite que el generador explore más antes de ser penalizado.  
                - Las prendas son razonablemente nítidas y variadas.
                """
            )
else:
    st.info("Usa el botón de la barra lateral para generar imágenes. 🎨")

st.write("---")

st.markdown(
    """
    ### 📌 Notas técnicas

    - Modelo: **DCGAN** entrenado sobre Fashion-MNIST (28x28, escala de grises).  
    - Hiperparámetros principales:
        - Vector de ruido `z` de dimensión 100.  
        - Activación final `Tanh`, salida normalizada en [-1, 1].  
    - La app está pensada como **complemento visual** del cuaderno de entrenamiento,
      donde se incluyen las métricas (realismo, diversidad, IS proxy) y los análisis
      de cada experimento.
    """
)
