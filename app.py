import streamlit as st
import numpy as np
from PIL import Image
from scipy.spatial import distance
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.preprocessing.image import img_to_array

# Configuration de la page
st.set_page_config(
    page_title="CBIR - Recherche d'Images",
    page_icon="🖼️",
    layout="wide",
)

# En-tête stylisé
st.markdown("""
<div style="background-color:#e8f0fe;padding:1rem;border-radius:10px;margin-bottom:1rem;">
    <h1 style="text-align:center;">🔍 Système CBIR</h1>
    <p style="text-align:center;">Recherche d'Images Basée sur le Contenu avec IA (VGG16, ResNet50, InceptionV3)</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Paramètres")
model_name = st.sidebar.selectbox("🧠 Modèle", ["VGG16", "ResNet50", "InceptionV3"])
metric = st.sidebar.selectbox("📏 Métrique de distance", ["Euclidienne", "Manhattan", "Chebyshev", "Canberra"])

# Upload
st.markdown("### 📤 Téléversez une image requête")
uploaded_file = st.file_uploader("Formats supportés : JPG / PNG", type=["jpg", "jpeg", "png"])

# Traitement
if uploaded_file is not None:
    query_img = Image.open(uploaded_file).convert("RGB")
    st.image(query_img, caption="🖼️ Image requête", use_column_width=True)

    if st.button("🔎 Lancer la recherche"):
        st.info("📡 Chargement du modèle et traitement de l'image...")

        vec_path = f"data/{model_name}_vectors.npy"
        label_path = f"data/{model_name}_labels.npy"
        img_path = f"data/{model_name}_paths.npy"

        DB_VECTORS = np.load(vec_path)
        DB_LABELS = np.load(label_path)
        IMAGE_PATHS = np.load(img_path, allow_pickle=True)

        # Modèle
        if model_name == "VGG16":
            model = VGG16(weights="imagenet", include_top=False, pooling="avg")
            size = (224, 224)
            preprocess = preprocess_vgg
        elif model_name == "ResNet50":
            model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
            size = (224, 224)
            preprocess = preprocess_resnet
        elif model_name == "InceptionV3":
            model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
            size = (299, 299)
            preprocess = preprocess_inception

        # Prétraitement
        img = query_img.resize(size)
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess(arr)
        query_vec = model.predict(arr).flatten()

        # Distance
        def compute_distance(vec1, vec2, metric):
            if metric == "Euclidienne":
                return distance.euclidean(vec1, vec2)
            elif metric == "Manhattan":
                return distance.cityblock(vec1, vec2)
            elif metric == "Chebyshev":
                return distance.chebyshev(vec1, vec2)
            elif metric == "Canberra":
                return distance.canberra(vec1, vec2)

        distances = [compute_distance(query_vec, vec, metric) for vec in DB_VECTORS]
        top_indices = np.argsort(distances)[:5]

        # Résultats
        st.subheader("🏆 Images les plus similaires")
        cols = st.columns(5)
        for i, idx in enumerate(top_indices):
            with cols[i]:
                st.image(IMAGE_PATHS[idx], caption=f"Label {DB_LABELS[idx]}\nDist: {distances[idx]:.2f}", use_column_width=True)

        # Histogramme
        st.subheader("📊 Répartition des labels")
        top_labels = [DB_LABELS[i] for i in top_indices]
        unique, counts = np.unique(top_labels, return_counts=True)

        fig, ax = plt.subplots()
        ax.bar(unique, counts, color="#4a90e2")
        ax.set_xlabel("Label")
        ax.set_ylabel("Fréquence")
        ax.set_title("Distribution des labels")
        st.pyplot(fig)

        # Animations finales
        st.success("✅ Analyse terminée avec succès !")
        st.balloons()

