import streamlit as st
import numpy as np
import librosa
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Initialize StandardScaler
scaler = StandardScaler()

def process_audio_chunk(audio_data, model, sr=22050):
    # Extract features (MFCC)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    
    # Reshape the features and scale
    features = np.array(mfccs_processed).reshape(1, -1)
    scaler.fit(features)  # Fit the scaler on the chunk
    features_scaled = scaler.transform(features)
    
    # Reshape for model input
    features_scaled = np.expand_dims(features_scaled, axis=2)
    
    # Predict emotion
    predictions = model.predict(features_scaled)
    
    return predictions[0]

def evaluate_audio_file(audio_path, model, chunk_duration=4):
    # Load the audio file
    data, sr = librosa.load(audio_path, sr=22050)
    total_duration = librosa.get_duration(y=data, sr=sr)
    
    # Process in 4-second chunks
    chunk_scores = []
    for start in range(0, int(total_duration), chunk_duration):
        end = min(start + chunk_duration, int(total_duration))
        chunk_data = data[start * sr:end * sr]
        if len(chunk_data) > 0:
            chunk_scores.append(process_audio_chunk(chunk_data, model, sr))
    
    # Aggregate results
    avg_scores = np.mean(chunk_scores, axis=0)
    emotions = ['angry', 'calm', 'disgust', 'fear', 'sad', 'neutral', 'happy', 'surprise']
    
    # Calculate sincerity, enthusiasm, and negative emotion scores
    sincerity_emotions = {'calm', 'neutral', 'happy'}
    enthusiasm_emotions = {'happy', 'surprise'}
    negative_emotions = {'sad', 'disgust', 'fear'}
    
    sincerity_score = sum(avg_scores[emotions.index(emotion)] for emotion in sincerity_emotions)
    enthusiasm_score = sum(avg_scores[emotions.index(emotion)] for emotion in enthusiasm_emotions)
    negative_score = sum(avg_scores[emotions.index(emotion)] for emotion in negative_emotions)
    
    return avg_scores, sincerity_score, enthusiasm_score, negative_score, emotions

# Streamlit App
def main():
    st.title("ASR Dashboard")
    st.write("Upload an audio file.")
    
    # File upload
    audio_file = st.file_uploader("Upload your audio file", type=["wav"])
    if audio_file is not None:
        # Save the uploaded file temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_file.read())
        
        # Load model
        model_path = 'D://Internships//Test Project//mark_1//emotion-recognition.h5'
        model = load_model(model_path)
        
        # Process audio file
        avg_scores, sincerity_score, enthusiasm_score, negative_score, emotions = evaluate_audio_file("temp_audio.wav", model)
        
        # Display results
        st.subheader("Emotion Analysis Results")
        st.write(f"Overall Sincerity Score: {sincerity_score:.2f}")
        st.write(f"Overall Enthusiasm Score: {enthusiasm_score:.2f}")
        st.write(f"Overall Negative Emotion Score: {negative_score:.2f}")
        
        # Emotion Probabilities Pie Chart
        st.subheader("Emotion Probabilities")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(avg_scores, labels=emotions, autopct='%1.1f%%', startangle=140, 
               colors=plt.cm.viridis(np.linspace(0, 1, len(avg_scores))))
        ax.set_title("Overall Emotion Probabilities")
        st.pyplot(fig)
        
        # Sincerity, Enthusiasm, and Negative Emotion Pie Chart
        st.subheader("Sincerity, Enthusiasm, and Negative Emotions")
        fig, ax = plt.subplots(figsize=(5, 5))
        labels = ['Sincerity', 'Enthusiasm', 'Negative Emotions']
        scores = [sincerity_score, enthusiasm_score, negative_score]
        colors = ['skyblue', 'orange', 'red']
        ax.pie(scores, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title("Sincerity, Enthusiasm, and Negative Emotion Scores")
        st.pyplot(fig)
        
        # Audio playback
        st.subheader("Audio Playback")
        st.audio("temp_audio.wav", format="audio/wav")

if __name__ == "__main__":
    main()
