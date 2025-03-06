
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.shortcuts import render
import os
import numpy as np
import librosa
import joblib

# Load the model and scaler
MODEL_PATH = os.path.join('detection', 'models', 'svm_model.pkl')
SCALER_PATH = os.path.join('detection', 'models', 'scaler.pkl')

try:
    svm_classifier = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")

def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        print(f"Loading audio file: {audio_path}")
        audio_data, sr = librosa.load(audio_path, sr=None)
        print(f"Audio file loaded. Sample rate: {sr}, Length: {len(audio_data)} samples")
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        print(f"Extracted MFCCs shape: {mfccs.shape}")
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting MFCC features: {e}")
        return None

def index(request):
    return JsonResponse({'message': 'Deepfake Audio Detection API'})
@csrf_exempt
def predict_audio(request):
    if request.method == 'POST' and request.FILES.get('audio'):
        audio_file = request.FILES['audio']
        
        # Ensure it's a .wav file
        if not audio_file.name.endswith('.wav'):
            return JsonResponse({'error': 'Invalid file format. Please upload a .wav file.'})

        # Save the file temporarily
        audio_path = os.path.join('detection', 'temp_audio.wav')
        with open(audio_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)

        # Extract MFCC features
        mfcc_features = extract_mfcc_features(audio_path)
        if mfcc_features is None:
            return JsonResponse({'error': 'Failed to extract features from audio'})

        # Make prediction
        try:
            mfcc_features_scaled = scaler.transform([mfcc_features])
            prediction = svm_classifier.predict(mfcc_features_scaled)
            result = "genuine" if prediction[0] == 0 else "deepfake"
            return JsonResponse({'result': result})
        except Exception as e:
            return JsonResponse({'error': f'Prediction failed: {e}'})

    return JsonResponse({'error': 'Invalid request'})
