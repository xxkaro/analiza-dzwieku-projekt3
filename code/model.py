import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
from audio_analysis import extract_mfcc

AUDIO_DIR = "/Users/karolina/Desktop/uni/6_semestr/analiza-dzwieku/audio_set/train"
TEST_DIR = "/Users/karolina/Desktop/uni/6_semestr/analiza-dzwieku/audio_set/test"
MODEL_PATH = "model_data/classifier.pkl"
ENCODER_PATH = "model_data/label_encoder.pkl"

def load_dataset(audio_dir):
    """Load audio dataset and extract MFCC features"""
    mfcc_list = []
    labels = []
    
    for person_dir in os.listdir(audio_dir):
        person_path = os.path.join(audio_dir, person_dir)
        if os.path.isdir(person_path):
            label = person_dir  
            for filename in os.listdir(person_path):
                if filename.endswith(".wav"):
                    filepath = os.path.join(person_path, filename)
                    try:
                        signal, sr = librosa.load(filepath, sr=16000)
                        mfcc = extract_mfcc(signal, sr)
                        mfcc_list.append(mfcc)
                        labels.append(label)

                    except Exception as e:
                        print(f"Błąd przy pliku {filepath}: {e}")
    
    return mfcc_list, labels

def evaluate_model(test_dir, model_path, encoder_path):
    """Evaluate the trained model on test data"""
    try:
        clf = joblib.load(model_path)
        le = joblib.load(encoder_path)
    except Exception as e:
        print(f"Błąd przy ładowaniu modelu: {e}")
        return
    
    print(f"\nWczytywanie danych testowych z: {test_dir}")
    X_test, y_test = load_dataset(test_dir)
    
    if not X_test:
        print("Brak danych testowych!")
        return
    
    print(f"Załadowano {len(X_test)} próbek testowych")
    
    try:
        y_test_encoded = le.transform(y_test)
    except ValueError as e:
        print(f"Błąd kodowania etykiet testowych: {e}")
        return
    
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    print("\n" + "="*50)
    print("WYNIKI EWALUACJI")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nWYNIKI KLASYFIKACJI:")
    print("-" * 50)
    report = classification_report(y_test_encoded, y_pred, 
                                 target_names=le.classes_, 
                                 digits=4)
    print(report)
    
    print("\nMACIERZ POMYŁEK:")
    print("-" * 50)
    cm = confusion_matrix(y_test_encoded, y_pred)
    print("Klasy:", le.classes_)
    print(cm)
 

def train_model():
    print("Wczytywanie danych z:", AUDIO_DIR)
    X, y = load_dataset(AUDIO_DIR)

    if not X:
        print("Brak danych.")
        return

    print(f"\nZaładowano {len(X)} próbek treningowych")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("\nTrening...")
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X, y_encoded)

    print("Zapisuję model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

    print(f"Model zapisany jako {MODEL_PATH} i etykiety jako {ENCODER_PATH}.")

def main():
    print("Co chcesz zrobić?")
    print("1 - Trenować model")
    print("2 - Przeprowadzić ewaluację")
    print("3 - Trenować model33 i przeprowadzić ewaluację")
    choice = input("Wybierz 1, 2 lub 3: ")

    if choice == "1":
        train_model()

    elif choice == "2":
        if os.path.exists(TEST_DIR):
            evaluate_model(TEST_DIR, MODEL_PATH, ENCODER_PATH)
        else:
            print(f"\nKatalog {TEST_DIR} nie istnieje.")

    elif choice == "3":
        train_model()
        if os.path.exists(TEST_DIR):
            evaluate_model(TEST_DIR, MODEL_PATH, ENCODER_PATH)
        else:
            print(f"\nKatalog {TEST_DIR} nie istnieje.")

    else:
        print("Nieprawidłowy wybór.")

if __name__ == "__main__":
    main()