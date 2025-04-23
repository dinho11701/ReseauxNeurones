# Développement d’un système de Récupération d'images 

## 📌 Description
Ce projet propose une application complète de recherche d’images similaires basée sur le contenu visuel (CBIR), sans recours à des annotations textuelles. Il exploite des modèles de réseaux de neurones convolutionnels pré-entraînés pour l’extraction de caractéristiques et applique l’algorithme de clustering K-Means pour organiser les images en groupes visuellement cohérents.

L’interface Web développée avec Streamlit permet aux utilisateurs de téléverser une image requête, de choisir un modèle et une métrique de distance, et d’obtenir les images les plus similaires accompagnées d’une analyse visuelle des résultats.

## 🧱 Architecture
Le système se décompose en deux phases :

### 1. Phase Offline
- Extraction des caractéristiques avec VGG16, ResNet50 et InceptionV3.
- Clustering par K-Means.
- Sauvegarde des vecteurs, labels et chemins des images dans des fichiers `.npy`.

### 2. Phase Online (App Web)
- Interface utilisateur avec Streamlit.
- Upload d’image requête et extraction des caractéristiques.
- Comparaison avec la base offline.
- Affichage des images similaires.
- Histogramme de répartition des clusters.

## 🚀 Installation

### Prérequis
- Python ≥ 3.10

### Clonage du projet
```bash
git clone https://github.com/ton-utilisateur/ReseauxNeurones.git
cd ReseauxNeurones
```

### Création de l'environnement virtuel
```bash
python -m venv venv
source venv/bin/activate
```

### Installation des dépendances
```bash
pip install -r requirements.txt
```

## 🧪 Utilisation

### 1. Générer la base offline :
```bash
python offline.py
```
Ce script extrait les vecteurs et effectue le clustering. Les fichiers générés sont enregistrés dans le dossier `data/`.

### 2. Lancer l'application Web :
```bash
streamlit run app.py
```
Ouvrir le lien affiché dans le terminal (`http://localhost:8501`).

## 📂 Structure du projet
```
.
├── app.py                      # Interface utilisateur
├── offline.py                 # Script d'extraction + clustering
├── data/                      # Données offline (.npy)
├── Datasets/Datasets                  # Images d'entraînement
├── requirements.txt           # Fichier des dépendances
├── README.md                  # Documentation
└── venv/                      # Environnement virtuel (non versionné)
```

## 📊 Technologies
- Python
- TensorFlow / Keras
- scikit-learn (KMeans)
- NumPy / Pillow
- Streamlit
- Matplotlib
- tqdm

## 👤 Auteur
Projet réalisé dans le cadre du cours INF5071 – UQAM
Par Andy Kouassi et Oswald Essongué

---

**Date limite de remise : 22 avril 2025**
