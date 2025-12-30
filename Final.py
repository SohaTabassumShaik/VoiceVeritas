import ctypes
import streamlit as st
import pandas as pd
import numpy as np
import librosa
import librosa.display
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import sounddevice as sd
import speech_recognition as sr
from gtts import gTTS
import pygame
import os
from PIL import Image

# Function to control keyboard LEDs
def toggle_leds(state):
    """Toggle Caps Lock and Num Lock LEDs based on state."""
    user32 = ctypes.WinDLL("user32")
    VK_CAPITAL = 0x14  # Caps Lock
    VK_NUMLOCK = 0x90  # Num Lock

    # Get current states
    caps_state = user32.GetKeyState(VK_CAPITAL) & 1
    num_state = user32.GetKeyState(VK_NUMLOCK) & 1

    # Toggle to desired state
    if caps_state != state:
        user32.keybd_event(VK_CAPITAL, 0, 0, 0)
        user32.keybd_event(VK_CAPITAL, 0, 2, 0)

    if num_state != state:
        user32.keybd_event(VK_NUMLOCK, 0, 0, 0)
        user32.keybd_event(VK_NUMLOCK, 0, 2, 0)

# Load the dataset
def load_dataset():
    try:
        dataset = pd.read_csv("Book1.csv")
        dataset = dataset.drop(columns=["Unnamed: 0"], errors="ignore")
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Feature extraction function
def extract_features(signal, sample_rate):
    magnitude = np.abs(signal)
    power = np.square(signal)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)

    features = {
        "M_mean": np.mean(magnitude),
        "M_variance": np.var(magnitude),
        "M_skewness": skew(magnitude),
        "M_kurtosis": kurtosis(magnitude),
        "P_mean": np.mean(power),
        "P_variance": np.var(power),
        "P_skewness": skew(power),
        "P_kurtosis": kurtosis(power),
        "MFCC_Mean": np.mean(mfcc),
        "MFCC_Variance": np.var(mfcc),
        "Delta_Mean": np.mean(librosa.feature.delta(mfcc)),
        "Delta_Variance": np.var(librosa.feature.delta(mfcc)),
        "DoubleDelta_Mean": np.mean(librosa.feature.delta(mfcc, order=2)),
        "DoubleDelta_Variance": np.var(librosa.feature.delta(mfcc, order=2))
    }
    return pd.DataFrame([features])

# Train classifier
def train_classifier(dataset):
    try:
        scaler = RobustScaler()
        le = LabelEncoder()
        dataset.iloc[:, :-1] = scaler.fit_transform(dataset.iloc[:, :-1])
        dataset["Class"] = le.fit_transform(dataset["Class"])

        clf = AdaBoostClassifier(n_estimators=50, random_state=89, learning_rate=0.5)
        clf.fit(dataset.iloc[:, :-1], dataset["Class"])
        return clf, scaler, le
    except Exception as e:
        st.error(f"Error training classifier: {e}")
        return None, None, None

# Speech recognition function
def recognize_speech_from_audio(signal, sample_rate):
    recognizer = sr.Recognizer()
    try:
        # Convert signal to bytes for speech recognition
        audio_data = sr.AudioData(np.int16(signal * 32768).tobytes(), sample_rate, 2)
        text = recognizer.recognize_google(audio_data)
        st.success(f"Recognized Text: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Could not understand the audio.")
        play_failure_audio_feedback("Apologies! I didn't understand. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"Speech Recognition Error: {e}")
        return None

# Provide audio feedback using gTTS and pygame
def play_audio_feedback(text):
    try:
        # Generate audio file
        tts = gTTS(text=text, lang='en')
        audio_file = "feedback.mp3"
        tts.save(audio_file)

        # Play the audio
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            continue

        # Clean up
        pygame.mixer.quit()
        os.remove(audio_file)  # Delete the audio file after playing
    except Exception as e:
        st.error(f"Error providing audio feedback: {e}")

# Provide failure audio feedback
def play_failure_audio_feedback(text):
    try:
        # Generate audio file
        tts = gTTS(text=text, lang='en')
        audio_file = "failure_feedback.mp3"
        tts.save(audio_file)

        # Play the audio
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            continue

        # Clean up
        pygame.mixer.quit()
        os.remove(audio_file)  # Delete the audio file after playing
    except Exception as e:
        st.error(f"Error providing failure audio feedback: {e}")

# Streamlit App
st.title("Voice Veritas: Truth in Voices")

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Start Recording to classify real-time audio.
2. View the classification result dynamically.
3. Explore the dataset to view its structure and statistics.
""")
section = st.sidebar.radio("Go to", ["Home", "Classify Voice", "Explore Dataset"])

if section == "Home":
    st.subheader("Welcome")
    st.write("Use this dashboard to classify audio as human or AI-generated and explore key features.")

    # Load and resize the image
    img = Image.open("aiandhuman.jpg")
    resized_img = img.resize((300, 300))  # Resize the image to 400x400 pixels (adjust as needed)
    # Add custom CSS to center the image
    st.markdown(
        """
        <style>
        .centered-image {
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns
    with col2:  # Place the image in the middle column
        st.image(resized_img, caption="AI VS Human")

elif section == "Classify Voice":
    st.subheader("Classify Voice")

    if st.button("Record Audio"):
        toggle_leds(state=0)  # Ensure LEDs are off before starting a new recording
        st.info("Recording... Speak now!")

        progress_placeholder = st.empty()
        progress_bar = st.empty()

        def update_progress(percentage, message):
            with progress_placeholder.container():
                st.markdown(f"{message} ({percentage}%)")
                progress_bar.progress(percentage)

        update_progress(0, "Initializing...")

        duration = 5  # seconds
        sample_rate = 22050
        try:
            update_progress(10, "Recording audio")
            signal = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()
            signal = signal.flatten()
            st.success("Recording complete!")
            update_progress(30, "Processing audio")

            recognized_text = recognize_speech_from_audio(signal, sample_rate)
            if recognized_text is None:
                st.warning("Speech recognition failed. Unable to process further.")
                st.stop()  # Streamlit's way to stop execution

            update_progress(50, "Extracting features")

            features_df = extract_features(signal, sample_rate)
            st.write("Extracted Features:", features_df)
            update_progress(70, "Loading dataset")

            dataset = load_dataset()
            if dataset is not None:
                update_progress(80, "Training classifier")
                clf, scaler, le = train_classifier(dataset)
                if clf is not None:
                    update_progress(90, "Making predictions")
                    scaled_features = scaler.transform(features_df)
                    prediction = clf.predict(scaled_features)
                    predicted_label = le.inverse_transform(prediction)[0]
                    st.success(f"Prediction: {predicted_label}")
                    update_progress(100, "Completed")

                    # LED Control Logic
                    if predicted_label.lower() == "human":
                        toggle_leds(state=1)
                        st.info("Caps Lock and Num Lock LEDs are ON")
                        audio_feedback = "The voice is generated by a Human, Proceed"
                    else:
                        toggle_leds(state=0)
                        st.info("Caps Lock and Num Lock LEDs are OFF")
                        audio_feedback = "The voice is AI generated, Access Denied"

                    play_audio_feedback(audio_feedback)
        except Exception as e:
            st.error(f"Error during audio recording or processing: {e}")
            update_progress(0, "Error encountered. Please try again.")


elif section == "Explore Dataset":
    st.subheader("Dataset Viewer")
    dataset = load_dataset()
    if dataset is not None:
        st.write(dataset)

        # Dataset statistics
        st.write("Dataset Summary:")
        st.write(dataset.describe())

        # Visualize dataset
        st.write("Class Distribution:")
        fig, ax = plt.subplots()
        dataset["Class"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)
