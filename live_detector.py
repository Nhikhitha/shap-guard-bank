import speech_recognition as sr
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# --- 1. One-time NLTK downloads ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK wordnet...")
    nltk.download('wordnet')

# --- 2. Define the Preprocessing Function ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))  # Convert to string and clean
    text = text.lower()                   
    text = text.split()                   
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    text = ' '.join(text)
    return text

# --- 3. Load the Saved Model and Vectorizer ---
model_path = 'spam_model.joblib'
vectorizer_path = 'vectorizer.joblib'

print("Loading trained model and vectorizer...")
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    print("\n--- 🚨 ERROR 🚨 ---")
    print("Model files not found! Make sure 'spam_model.joblib' and 'vectorizer.joblib'")
    print("are in the same folder as this script.")
    print("--------------------")
    exit()

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
print("✅ Model loaded. Ready to detect spam.")

# --- 4. Initialize the Speech Recognizer ---
recognizer = sr.Recognizer()
# ---  THIS IS YOUR CHANGE ---
recognizer.pause_threshold = 3.0  # Waits for 3 seconds of silence
# ------------------------------
microphone = sr.Microphone()

print("\n(A one-time check for your microphone...)")
with microphone as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)
# ---  I UPDATED THIS PRINT STATEMENT ---
print(f"Microphone check complete. Will process after {recognizer.pause_threshold} sec of silence.")
# -----------------------------------

# --- 5. Main Loop: Listen, Transcribe, and Predict ---
if __name__ == "__main__":
    while True:
        print("\nPress ENTER to start speaking...")
        try:
            input() # This line will WAIT until you press Enter
        except KeyboardInterrupt:
            print("\nStopping detector.")
            break

        print("🎤 Listening... Speak your full sentence now.")
        # ---  I UPDATED THIS PRINT STATEMENT ---
        print(f"(It will wait for {recognizer.pause_threshold}s of silence, then process)")
        
        with microphone as source:
            try:
                audio = recognizer.listen(source, timeout=5) 
                
                print("Processing audio...")
                text = recognizer.recognize_google(audio)
                
                print(f"You said: '{text}'")

                # --- PREDICTION ---
                if not text.strip():
                    print("No speech detected. Listening again.")
                    continue

                processed_text = preprocess_text(text)
                text_vector = vectorizer.transform([processed_text])
                prediction = model.predict(text_vector)
                probability = model.predict_proba(text_vector)
                
                # --- Show Result ---
                if prediction[0] == 1: # 1 means SCAM
                    fraud_prob = probability[0][1] * 100
                    print(f"--- 🚨 ALERT: SCAM DETECTED! 🚨 --- (Confidence: {fraud_prob:.2f}%)")
                else: # 0 means NOT SCAM
                    normal_prob = probability[0][0] * 100
                    print(f"--- ✅ NORMAL call --- (Confidence: {normal_prob:.2f}%)")

            except sr.WaitTimeoutError:
                print("You pressed Enter but didn't speak. Please try again.")
            except sr.UnknownValueError:
                print("Could not understand audio. Please try again.")
            except sr.RequestError as e:
                print(f"Google Speech Recognition service error; {e}")
            except KeyboardInterrupt:
                print("\nStopping detector.")
                break
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break