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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible;}
    </style>
    """,
    unsafe_allow_html=True
)

# Directorio donde están los modelos
MODELS_DIR = os.path.join(os.path.dirname(__file__), "modelos")

# Hiperparámetros (DEBEN coincidir con el notebook)
nz = 100   # tamaño vector latente
nc = 1     # canales (escala de grises)
ngf = 32   # IMPORTANTE: debe ser 32 como en el notebook
ndf = 32

# ============================
# Definición del Generador
# (EXACTAMENTE igual al del notebook)
# ============================

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input: Z (nz x 1 x 1)
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),   # 4x4
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),      # 32x32
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),           # 64x64
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)

# ============================
# Detectar modelos disponibles
# ============================

def obtener_modelos_disponibles():
    """Detecta qué modelos .pth existen en la carpeta modelos/"""
    todos_los_experimentos = {
        "Experimento 1 – Baseline": "exp1_baseline.pth",
        "Experimento 2 – Más épocas": "exp2_mas_epocas.pth",
        "Experimento 3 – lrD más bajo": "exp3_lrD_bajo.pth",
    }
    
    disponibles = {}
    for nombre, archivo in todos_los_experimentos.items():
        ruta = os.path.join(MODELS_DIR, archivo)
        if os.path.exists(ruta):
            disponibles[nombre] = archivo
    
    return disponibles

# ============================
# Funciones auxiliares
# ============================

@st.cache_resource(show_spinner=False)
def cargar_modelo(nombre_archivo: str):
    """Carga un modelo de generador desde modelos/*.pth"""
    ruta = os.path.join(MODELS_DIR, nombre_archivo)
    if not os.path.exists(ruta):
        return None, f"No se encontró el archivo: {ruta}"

    device = torch.device("cpu")
    model = Generator().to(device)
    
    try:
        state = torch.load(ruta, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Error al cargar el modelo: {str(e)}"

def generar_imagenes(modelo: nn.Module, num_imagenes: int, seed: int):
    """Genera imágenes sintéticas con el generador entrenado"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cpu")
    noise = torch.randn(num_imagenes, nz, 1, 1, device=device)
    
    with torch.no_grad():
        fake = modelo(noise).detach().cpu()

    # Normalizar de [-1, 1] a [0, 1]
    fake = (fake + 1) / 2
    
    # Crear grid de imágenes
    grid = make_grid(fake, nrow=int(np.sqrt(num_imagenes)), padding=2)
    ndarr = grid.mul(255).clamp(0, 255).byte().numpy()

    # Convertir a imagen PIL
    if ndarr.shape[0] == 1:
        ndarr = ndarr[0]  # [H, W]
        img = Image.fromarray(ndarr, mode="L")
    else:
        ndarr = np.transpose(ndarr, (1, 2, 0))
        img = Image.fromarray(ndarr)

    return img

# ============================
# Layout de la aplicación
# ============================

st.title("🧥 Generación de ropa con DCGAN – Deep Learning Avanzado")
st.caption("Proyecto de IA generativa con Fashion-MNIST | IUDigital")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        Esta aplicación permite **generar prendas de ropa sintéticas** a partir de un modelo
        **DCGAN** (Deep Convolutional GAN) entrenado sobre el dataset Fashion-MNIST.

        Puedes seleccionar distintos **experimentos** y comparar la calidad de las imágenes generadas:
        - **Baseline**: configuración estándar (10 épocas)
        - **Más épocas**: entrenamiento extendido (20 épocas)
        - **lrD bajo**: tasa de aprendizaje reducida en el discriminador
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

# ============================
# Verificar modelos disponibles
# ============================

experimentos_disponibles = obtener_modelos_disponibles()

if not experimentos_disponibles:
    st.error(
        """
        ⚠️ **No se encontraron modelos entrenados**
        
        La carpeta `modelos/` está vacía o no contiene archivos `.pth`.
        
        **Para generar los modelos:**
        1. Ejecuta el notebook completo `IAGENERATIVA_DEEPLEARNING.ipynb`
        2. Descarga los archivos `.pth` generados
        3. Súbelos a la carpeta `modelos/` en este repositorio
        
        **Archivos necesarios:**
        - `exp1_baseline.pth`
        - `exp2_mas_epocas.pth`
        - `exp3_lrD_bajo.pth`
        """,
        icon="🚫"
    )
    st.stop()

# Mostrar advertencia si faltan modelos
todos_los_modelos = 3
modelos_disponibles = len(experimentos_disponibles)

if modelos_disponibles < todos_los_modelos:
    st.warning(
        f"⚠️ Solo **{modelos_disponibles} de {todos_los_modelos}** experimentos están disponibles. "
        f"Ejecuta el notebook completo para generar todos los modelos.",
        icon="⚡"
    )

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Controles de generación")

exp_nombre = st.sidebar.selectbox(
    "Modelo a utilizar",
    list(experimentos_disponibles.keys()),
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
    help="Cambia este número para obtener diferentes resultados"
)

auto_generar = st.sidebar.checkbox(
    "Generar automáticamente al cambiar parámetros",
    value=True,
)

btn_generar = st.sidebar.button("🎨 Generar imágenes", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    **Estado de los modelos**
    
    ✅ Disponibles: **{modelos_disponibles}/{todos_los_modelos}**
    
    **Descripción de experimentos:**

    - **Exp 1 (Baseline)**: DCGAN estándar, 10 épocas, lr=0.0002
    - **Exp 2 (Más épocas)**: 20 épocas de entrenamiento
    - **Exp 3 (lrD bajo)**: Discriminador con lr=0.0001
    """
)

# ============================
# Zona principal de resultados
# ============================

st.subheader(f"📸 Imágenes generadas – {exp_nombre}")

modelo, error = cargar_modelo(experimentos_disponibles[exp_nombre])

if error:
    st.error(
        f"""
        **Error al cargar el modelo:**
        
        {error}
        
        **Posibles causas:**
        - El archivo `.pth` está corrupto
        - La arquitectura del modelo no coincide
        - El archivo fue generado con una versión diferente de PyTorch
        
        **Solución:** Regenera el modelo ejecutando el notebook nuevamente.
        """,
        icon="⚠️"
    )
    st.stop()

# Lógica de generación
debe_generar = auto_generar or btn_generar

if debe_generar:
    with st.spinner("🎨 Generando imágenes sintéticas..."):
        try:
            img = generar_imagenes(modelo, num_imgs, seed)
            
            st.image(
                img, 
                caption=f"{num_imgs} prendas sintéticas generadas con {exp_nombre}",
                use_container_width=True
            )
            
            st.success(f"✅ Se generaron {num_imgs} imágenes exitosamente", icon="✨")
            
        except Exception as e:
            st.error(f"Error al generar imágenes: {str(e)}", icon="❌")

    # Detalles del experimento
    with st.expander("🧪 Detalles del experimento seleccionado"):
        if "Baseline" in exp_nombre:
            st.markdown(
                """
                ### Experimento 1 – Baseline

                **Configuración:**
                - Arquitectura: DCGAN estándar
                - Épocas: 10
                - Learning rate G: 0.0002
                - Learning rate D: 0.0002
                - Optimizador: Adam (β₁=0.5)

                **Resultados:**
                - Buen equilibrio entre realismo y estabilidad
                - Algunas prendas pueden ser borrosas
                - Diversidad aceptable en los resultados
                
                **Uso recomendado:** Punto de partida para comparación
                """
            )
        elif "Más épocas" in exp_nombre:
            st.markdown(
                """
                ### Experimento 2 – Más épocas

                **Configuración:**
                - Arquitectura: DCGAN estándar
                - Épocas: 20 (2x baseline)
                - Learning rate G: 0.0002
                - Learning rate D: 0.0002
                - Optimizador: Adam (β₁=0.5)

                **Resultados:**
                - Mayor nitidez y definición en las prendas
                - Mejor captura de detalles finos
                - Ligero riesgo de sobreajuste
                - Mantiene buena diversidad
                
                **Uso recomendado:** Producción de imágenes de alta calidad
                """
            )
        else:
            st.markdown(
                """
                ### Experimento 3 – lrD más bajo

                **Configuración:**
                - Arquitectura: DCGAN estándar
                - Épocas: 20
                - Learning rate G: 0.0002
                - Learning rate D: 0.0001 (50% del baseline)
                - Optimizador: Adam (β₁=0.5)

                **Resultados:**
                - Balance mejorado entre G y D
                - El generador tiene más oportunidad de aprender
                - Prendas nítidas y variadas
                - Entrenamiento más estable
                
                **Uso recomendado:** Exploración de variaciones creativas
                """
            )
else:
    st.info("👈 Usa los controles de la barra lateral para generar imágenes", icon="💡")

st.write("---")

# Información técnica
with st.expander("📚 Información técnica del proyecto"):
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(
            """
            ### Arquitectura DCGAN
            
            **Generador:**
            - Input: Vector latente z (100 dim)
            - 5 capas ConvTranspose2d
            - BatchNorm + LeakyReLU
            - Output: 64×64 escala de grises
            - Activación final: Tanh
            
            **Discriminador:**
            - Input: Imagen 64×64
            - 5 capas Conv2d
            - BatchNorm + LeakyReLU
            - Output: Probabilidad [0,1]
            - Activación final: Sigmoid
            """
        )
    
    with col_b:
        st.markdown(
            """
            ### Dataset y métricas
            
            **Fashion-MNIST:**
            - 60,000 imágenes de entrenamiento
            - 10 categorías de ropa
            - Resolución: 28×28 (escalado a 64×64)
            - Escala de grises
            
            **Métricas evaluadas:**
            - Realism Score (clasificador)
            - Diversity Score (variedad)
            - IS Proxy (Inception Score)
            """
        )

st.markdown(
    """
    ---
    ### 🎯 Casos de uso prácticos
    
    1. **Diseño de moda:** Generación de bocetos iniciales para nuevas colecciones
    2. **E-commerce:** Creación de variaciones de productos para testing A/B
    3. **Data augmentation:** Aumento de datos para entrenar clasificadores de ropa
    4. **Educación:** Demostración de conceptos de IA generativa
    
    ---
    
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    Proyecto desarrollado con PyTorch y Streamlit | Deep Learning Avanzado 2024
    </div>
    """,
    unsafe_allow_html=True
)