import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm

# Dossier contenant les images
IMAGE_DIR = "Datasets/Datasets/"
image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]

# Initialiser les modèles
MODELS = {
    "VGG16": (VGG16(weights="imagenet", include_top=False, pooling="avg"), (224, 224), preprocess_vgg),
    "ResNet50": (ResNet50(weights="imagenet", include_top=False, pooling="avg"), (224, 224), preprocess_resnet),
    "InceptionV3": (InceptionV3(weights="imagenet", include_top=False, pooling="avg"), (299, 299), preprocess_inception),
}

# Pour chaque modèle, on sauvegarde les résultats
for model_name, (model, target_size, preprocess) in MODELS.items():
    print(f"📥 Traitement avec {model_name}...")

    features = []
    valid_paths = []

    for path in tqdm(image_paths):
        try:
            img = load_img(path, target_size=target_size)
            arr = img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess(arr)
            vec = model.predict(arr).flatten()
            features.append(vec)
            valid_paths.append(path)
        except Exception as e:
            print(f"Erreur pour {path} : {e}")

    features = np.array(features)
    np.save(f"{model_name}_vectors.npy", features)
    np.save(f"{model_name}_paths.npy", valid_paths)

    # Clustering avec KMeans (nombre de clusters = 5 pour l’exemple)
    print(f"🔍 Clustering avec KMeans pour {model_name}...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(features)
    np.save(f"{model_name}_labels.npy", labels)

    print(f"✅ Données sauvegardées pour {model_name}")

