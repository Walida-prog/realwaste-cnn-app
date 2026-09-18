import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Tri des Déchets - CNN RealWaste",
    page_icon="♻️",
    layout="centered"
)

# Liste exacte des 9 classes issues du dataset (ordre alphabétique découvert par Keras)
CLASS_NAMES = [
    'Cardboard', 'Food Waste', 'Glass', 'Metal', 
    'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile', 'Vegetation'
]

IMG_SIZE = 128

@st.cache_resource
def load_model():
    """Charge le modèle en mémoire une seule fois au démarrage."""
    return tf.keras.models.load_model("modele_dechets.keras")

# Chargement du modèle
with st.spinner("Chargement du modèle CNN en mémoire..."):
    model = load_model()

# Titre et interface
st.title("♻️ Classification Automatique de Déchets")
st.write("Téléversez la photo d'un déchet pour identifier sa catégorie parmi les 9 classes de **RealWaste**.")

# Widget d'import d'image
uploaded_file = st.file_uploader(
    "Choisissez une image (.jpg, .jpeg, .png)", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # 1. Lecture et affichage de l'image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image importée", use_container_width=True)

    # 2. Prétraitement conforme à l'entraînement
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)  # Forme : (1, 128, 128, 3)

    # 3. Inférence
    predictions = model.predict(img_batch, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = predictions[predicted_idx] * 100

    # 4. Affichage des résultats
    st.markdown("---")
    st.subheader("Résultat de l'analyse :")
    st.success(f"**Classe prédite :** {predicted_class} ({confidence:.2f}%)")

    # 5. Graphique de répartition des probabilités sur les 9 classes
    st.write("Détail des probabilités :")
    prob_dict = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
    st.bar_chart(prob_dict)