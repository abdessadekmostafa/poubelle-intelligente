import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2

@st.cache_data()
def load_model_cache():
    model_path = "deuxieme_modele.keras"
    model = load_model(model_path, compile=False)
    return model

# Chargement du modèle
model = load_model_cache()

def preprocess_image(upload):
    try:
        img = Image.open(upload).convert("RGB")  # Conversion en format RGB
        img = np.asarray(img)
        img_resize = cv2.resize(img, (224, 224)) / 255.0  # Normalisation
        img_resize = np.expand_dims(img_resize, axis=0)  # Ajout d'une dimension batch
        return img_resize
    except Exception as e:
        st.error(f"Erreur de traitement de l'image : {e}")
        return None

def predict(upload):
    img_preprocessed = preprocess_image(upload)
    if img_preprocessed is not None:
        pred = model.predict(img_preprocessed)
        return pred[0][0]  # Probabilité que l'objet soit recyclable
    else:
        return None

# Interface utilisateur
st.title("Poubelle Intelligente")
st.write("Glissez et déposez une image d'objet ci-dessous pour savoir si elle est recyclable ou non.")

# Gestion du glisser-déposer
upload = st.file_uploader("Chargez ou déposez une image", type=['png', 'jpeg', 'jpg', 'webp'])

threshold = 50 #st.slider("Définissez le seuil pour la recyclabilité (%)", 0, 100, 50)

if upload:
    rec = predict(upload)
    if rec is not None:
        prob_recyclable = rec * 100      
        prob_organic = (1 - rec) * 100

        st.image(Image.open(upload), caption="Image chargée", use_column_width=True)
        if prob_recyclable > threshold:
            st.success(f"✅ L'objet est recyclable avec une certitude de {prob_recyclable:.2f} %")
        else:
            st.error(f"❌ L'objet n'est pas recyclable avec une certitude de {prob_organic:.2f} %")
    else:
        st.write("❗ Le modèle n'a pas pu traiter l'image téléchargée.")
else:
    st.info("Glissez ou déposez une image pour commencer.")

