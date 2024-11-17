import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Configurar el modelo desde Hugging Face
MODEL_NAME = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# Crear interfaz de usuario
st.set_page_config(page_title="Identificador de Ropa con AI", page_icon="👚", layout="centered")
st.title("🔍 Identificador de Prendas de Ropa con IA")
st.markdown("### ¡Sube una imagen y te diremos qué prenda es! 😄")
st.write("Usa esta herramienta para identificar ropa en tus imágenes.")

# Subida de imagen
uploaded_file = st.file_uploader("📸 Sube una imagen de ropa (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Mostrar la imagen cargada con un tamaño más pequeño
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_column_width=True, width=400)  # Limitar el tamaño de la imagen
    
    # Mostrar un mensaje mientras se procesa la imagen
    st.write("Procesando imagen... 🧠")

    # Etiquetas ampliadas (eliminamos "pijama")
    labels = [
        "camisa", "vestido", "pantalón", "abrigo", "zapatos",
        "chaqueta", "traje", "corbata",
        "blusa", "chaleco", "camiseta",
        "zapatos"
    ]
    
    # Preprocesar la imagen y hacer inferencia
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

    with st.spinner("Realizando la inferencia... 🔍"):
        # Inferencia con el modelo
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)  # Convertir logits a probabilidades

    # Filtrar etiquetas con un porcentaje mayor al 30%
    threshold = 0.30  # 30% 
    filtered_labels = [(label, prob.item()) for label, prob in zip(labels, probs[0]) if prob.item() > threshold]

    # Mostrar las etiquetas filtradas
    st.subheader("🔎 Resultados de la predicción")

    if filtered_labels:
        for label, prob in filtered_labels:
            st.markdown(f"**{label.capitalize()}**: {prob*100:.2f}%")
    else:
        # Si no hay etiquetas mayores al 30%, mostrar la de mayor probabilidad
        max_label, max_prob = max(zip(labels, probs[0]), key=lambda x: x[1].item())
        st.markdown(f"**La prenda más probable es: {max_label.capitalize()}** con un **{max_prob*100:.2f}%** de probabilidad.")
        
        # Asegurarse de que max_label no sea vacío o nulo antes de mostrar el HTML
        if max_label:
            st.markdown(f"<h2 style='text-align: center; color: #1e90ff;'>{max_label.capitalize()}</h2>", unsafe_allow_html=True)

    # Espacio adicional
    st.markdown("---")
    st.write("¡Prueba con otra imagen para ver más resultados! 🌟")
