# 🖼️ Développement d’un système de Récupération d'images (CBIR)

## 📌 Description

Ce projet propose une application complète de **recherche d’images similaires basée sur le contenu visuel** (CBIR), sans recours à des annotations textuelles.

Il s'appuie sur :
- des modèles de réseaux de neurones convolutionnels pré-entraînés pour **l’extraction de caractéristiques** ;
- l’algorithme de **clustering K-Means** pour organiser les images en groupes visuellement cohérents.

L’interface Web développée avec **Streamlit** permet aux utilisateurs de :
- téléverser une image requête ;
- choisir un modèle et une métrique de distance ;
- obtenir les images les plus similaires avec **une visualisation intuitive des résultats**.

---

## 🧱 Architecture

Le système se divise en **deux phases** :

### 🔧 1. Phase Offline

- Extraction des caractéristiques avec `VGG16`, `ResNet50`, `InceptionV3`.
- Clustering avec `KMeans`.
- Sauvegarde des vecteurs, labels et chemins d'images dans des fichiers `.npy` dans un dossier `data/` (à créer manuellement).

### 🌐 2. Phase Online (Interface Web)

- Interface utilisateur avec **Streamlit**.
- Upload d’image requête.
- Extraction des caractéristiques.
- Comparaison avec la base offline.
- Affichage des images similaires + histogramme des labels.

---

## 🚀 Installation

### ✅ Prérequis

- Python ≥ **3.10**

### 📥 Clonage du projet

```bash
git clone https://github.com/ton-utilisateur/ReseauxNeurones.git
cd ReseauxNeurones
```

### 🧪 Création de l’environnement virtuel

```bash
python -m venv venv
source venv/bin/activate
```

### 📦 Installation des dépendances

```bash
pip install -r requirements.txt
```

### 📂 Préparation des données

1. **Créez manuellement un dossier `data/` à la racine du projet**.
2. **Téléchargez le jeu d'images du TP** et placez-le dans :

```
Datasets/Datasets/
```

---

## 🧪 Utilisation

### 1️⃣ Générer les données offline (vecteurs + clustering) :

```bash
python offline.py
```

> Les fichiers `.npy` seront générés dans le dossier `data/`.

### 2️⃣ Lancer l’application web :

```bash
streamlit run app.py
```

> Ouvrez ensuite le lien dans le terminal (`http://localhost:8501`).

---

## 📂 Structure du projet

```
.
├── app.py                  # Interface Streamlit
├── offline.py              # Extraction de caractéristiques + clustering
├── data/                   # Données générées (.npy)
├── Datasets/Datasets/      # Jeu d’images à télécharger
├── requirements.txt        # Fichier des dépendances
├── README.md               # Documentation
└── venv/                   # Environnement virtuel Python
```

---

## 📊 Technologies utilisées

- Python
- TensorFlow / Keras
- scikit-learn (KMeans)
- NumPy / Pillow
- Streamlit
- Matplotlib
- tqdm

---

## 👨‍💻 Auteurs

Projet réalisé dans le cadre du cours **INF5071 – UQAM**  
Par **Andy Kouassi** et **Oswald Essongué**

---

**📅 Date limite de remise : 22 avril 2025**
