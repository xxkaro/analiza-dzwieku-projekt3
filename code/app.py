import streamlit as st
import librosa
import joblib
import os
from audio_analysis import extract_mfcc

MODEL_PATH = "model_data/classifier.pkl"
ENCODER_PATH = "model_data/label_encoder.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        clf = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        return clf, le
    return None, None

st.title("Identyfikacja mówcy")

clf, le = load_model()


if clf is None or le is None:
    st.error("Model nie został znaleziony! Upewnij się, że pliki modelu istnieją.")
    st.info(f"Szukane pliki: {MODEL_PATH}, {ENCODER_PATH}")
else:
    st.success(f"Model załadowany!")
    
    audio_file = st.file_uploader("Próbka do identyfikacji", type=["wav"])
    
    if audio_file:
        st.audio(audio_file, format="audio/wav")
        signal, sr = librosa.load(audio_file, sr=16000)
        mfcc_features = extract_mfcc(signal, sr)
        
        prediction = clf.predict([mfcc_features])[0]
        probabilities = clf.predict_proba([mfcc_features])[0]
        confidence = max(probabilities)

        predicted_person = le.inverse_transform([prediction])[0]
        
        st.subheader("Wynik identyfikacji:")
        st.write(f"**Rozpoznana osoba:** {predicted_person}")
        st.write(f"**Pewność:** {round(confidence * 100, 1)}%")
        
        st.subheader("Wszystkie prawdopodobieństwa:")
        for i, person in enumerate(le.classes_):
            prob = probabilities[le.transform([person])[0]]
            st.write(f"{person}: {round(prob * 100, 1)}%")
            