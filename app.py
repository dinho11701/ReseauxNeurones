import streamlit as st
import numpy as np
from PIL import Image
from scipy.spatial import distance
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.preprocessing.image import img_to_array

# --- Interface utilisateur ---
st.title("🔍 Recherche d'images similaires")

uploaded_file = st.file_uploader("📤 Téléversez une image requête", type=["jpg", "jpeg", "png"])

model_name = st.selectbox("🧠 Choisissez un descripteur :", ["VGG16", "ResNet50", "InceptionV3"])
metric = st.selectbox("📏 Choisissez une métrique :", ["Euclidienne", "Manhattan", "Chebyshev", "Canberra"])

if uploaded_file is not None:
    query_img = Image.open(uploaded_file).convert("RGB")
    st.image(query_img, caption="Image requête", use_column_width=True)

    if st.button("🔎 Lancer la recherche"):
        # Chargement des fichiers correspondant au modèle choisi
        vec_path = f"data/{model_name}_vectors.npy"
        label_path = f"data/{model_name}_labels.npy"
        img_path = f"data/{model_name}_paths.npy"

        DB_VECTORS = np.load(vec_path)
        DB_LABELS = np.load(label_path)
        IMAGE_PATHS = np.load(img_path, allow_pickle=True)

        # Chargement du modèle et du prétraitement
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

        # Prétraitement et extraction du vecteur requête
        img = query_img.resize(size)
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess(arr)
        query_vec = model.predict(arr).flatten()

        # Calcul des distances
        def compute_distance(vec1, vec2, metric):
            if metric == "Euclidienne":
                return distance.euclidean(vec1, vec2)
            elif metric == "Manhattan":
                return distance.cityblock(vec1, vec2)
            elif metric == "Chebyshev":
                return distance.chebyshev(vec1, vec2)
            elif metric == "Canberra":
                return distance.canberra(vec1, vec2)
            else:
                raise ValueError("Métrique non supportée.")

        distances = [compute_distance(query_vec, vec, metric) for vec in DB_VECTORS]
        top_indices = np.argsort(distances)[:5]

        # Affichage des résultats
        st.subheader("🖼️ Images les plus proches :")
        for i in top_indices:
            st.image(IMAGE_PATHS[i], caption=f"Label : {DB_LABELS[i]} — Distance : {distances[i]:.2f}", width=150)

        st.subheader("📊 Analyse des labels :")
        top_labels = [DB_LABELS[i] for i in top_indices]
        unique, counts = np.unique(top_labels, return_counts=True)

        fig, ax = plt.subplots()
        ax.bar(unique, counts)
        ax.set_xlabel("Label")
        ax.set_ylabel("Fréquence")
        ax.set_title("Répartition des labels parmi les plus proches")
        st.pyplot(fig)

        st.success("✅ Analyse terminée")

