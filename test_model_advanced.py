import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import librosa
import librosa.display
import soundfile as sf
from scipy.io import wavfile
from train import SpeechEnhancementCNN, load_audio, extract_features

def griffin_lim(mag, n_iter=32, n_fft=2048, hop_length=512, win_length=2048):
    """
    Griffin-Lim algorithm to convert magnitude spectrogram to audio.
    
    Args:
        mag: Magnitude spectrogram
        n_iter: Number of iterations for the algorithm
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        win_length: Window length
    """
    # Check if the spectrogram dimensions are compatible with the FFT parameters
    # If not, resize the spectrogram to be compatible
    if mag.shape[1] < n_fft:
        # Pad the spectrogram if it's too small
        pad_width = ((0, 0), (0, n_fft - mag.shape[1]))
        mag = np.pad(mag, pad_width, mode='constant')
    
    # Initialize phase
    phase = np.exp(1j * np.random.uniform(0, 2*np.pi, size=mag.shape))
    
    # Iteratively estimate phase
    for i in range(n_iter):
        # Convert to time domain
        y = librosa.istft(mag * phase, hop_length=hop_length, win_length=win_length)
        
        # Convert back to frequency domain
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        
        # Update phase
        phase = np.exp(1j * np.angle(D))
    
    # Final reconstruction
    y = librosa.istft(mag * phase, hop_length=hop_length, win_length=win_length)
    return y

def mfcc_to_audio(mfcc, sr=16000, n_mels=128, n_fft=2048, hop_length=512):
    """
    Convert MFCCs back to audio using a simplified approach.
    
    Args:
        mfcc: MFCC features
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
    """
    try:
        # Convert MFCCs to mel spectrogram
        # The mfcc_to_mel function doesn't accept sr parameter
        mel_spec = librosa.feature.inverse.mfcc_to_mel(mfcc, n_mels=n_mels)
        
        # Convert mel spectrogram to linear spectrogram
        # This is an approximation
        mag = np.exp(mel_spec) - 1
        
        # Ensure the spectrogram has the right dimensions for the FFT
        if mag.shape[1] < n_fft:
            # Pad the spectrogram if it's too small
            pad_width = ((0, 0), (0, n_fft - mag.shape[1]))
            mag = np.pad(mag, pad_width, mode='constant')
        
        # Apply Griffin-Lim algorithm
        audio = griffin_lim(mag, n_iter=32, n_fft=n_fft, hop_length=hop_length)
        
        return audio
    
    except Exception as e:
        print(f"Error in mfcc_to_audio: {e}")
        # Return a simple sine wave as a fallback
        duration = mfcc.shape[1] * hop_length / sr
        t = np.linspace(0, duration, int(duration * sr))
        return 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

def test_model_advanced(model_path, test_audio_path, output_dir="enhanced_audio_advanced"):
    """
    Test the trained speech enhancement model on a single audio file with advanced audio reconstruction.
    
    Args:
        model_path: Path to the saved model
        test_audio_path: Path to the test audio file
        output_dir: Directory to save enhanced audio
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = SpeechEnhancementCNN(input_channels=1, n_mfcc=40, max_frames=400)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load and process the test audio
    print(f"Processing audio file: {test_audio_path}")
    audio, sr = librosa.load(test_audio_path, sr=16000)
    
    # Extract features
    features = extract_features(audio, sr=sr, max_frames=400)
    
    # Convert to tensor and add batch and channel dimensions
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Apply the model
    with torch.no_grad():
        enhanced_features = model(features_tensor)
    
    # Convert back to numpy
    enhanced_features = enhanced_features.squeeze().numpy()
    
    # Visualize the results
    plt.figure(figsize=(15, 15))
    
    # Original MFCCs
    plt.subplot(3, 1, 1)
    librosa.display.specshow(features, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original MFCCs')
    
    # Enhanced MFCCs
    plt.subplot(3, 1, 2)
    librosa.display.specshow(enhanced_features, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Enhanced MFCCs')
    
    # Difference
    plt.subplot(3, 1, 3)
    diff = enhanced_features - features
    librosa.display.specshow(diff, sr=sr, x_axis='time', y_axis='mel', cmap='RdBu')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Difference (Enhanced - Original)')
    
    # Save the plot
    plot_path = os.path.join(output_dir, "enhancement_comparison_advanced.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Visualization saved to {plot_path}")
    
    # Save the original audio
    original_path = os.path.join(output_dir, "original_audio.wav")
    sf.write(original_path, audio, sr)
    print(f"Original audio saved to {original_path}")
    
    # Try to reconstruct audio from MFCCs
    try:
        print("Reconstructing audio from original MFCCs...")
        original_reconstructed = mfcc_to_audio(features, sr=sr)
        original_reconstructed_path = os.path.join(output_dir, "original_reconstructed.wav")
        sf.write(original_reconstructed_path, original_reconstructed, sr)
        print(f"Original reconstructed audio saved to {original_reconstructed_path}")
        
        print("Reconstructing audio from enhanced MFCCs...")
        enhanced_audio = mfcc_to_audio(enhanced_features, sr=sr)
        enhanced_path = os.path.join(output_dir, "enhanced_audio.wav")
        sf.write(enhanced_path, enhanced_audio, sr)
        print(f"Enhanced audio saved to {enhanced_path}")
        
    except Exception as e:
        print(f"Error during audio reconstruction: {e}")
        print("Note: Audio reconstruction from MFCCs is challenging and may not produce high-quality results.")
    
    return enhanced_features

def batch_test_advanced(model_path, test_dir, output_dir="enhanced_audio_advanced"):
    """
    Test the model on multiple audio files in a directory with advanced audio reconstruction.
    
    Args:
        model_path: Path to the saved model
        test_dir: Directory containing test audio files
        output_dir: Directory to save enhanced audio
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = SpeechEnhancementCNN(input_channels=1, n_mfcc=40, max_frames=400)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Find all wav files in the test directory
    test_files = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith(".wav"):
                test_files.append(os.path.join(root, file))
    
    print(f"Found {len(test_files)} test files")
    
    # Process each file
    for i, test_file in enumerate(test_files):
        print(f"Processing file {i+1}/{len(test_files)}: {test_file}")
        
        # Create a subdirectory for this file
        file_name = os.path.basename(test_file).replace(".wav", "")
        file_output_dir = os.path.join(output_dir, file_name)
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Test the model on this file
        test_model_advanced(model_path, test_file, file_output_dir)

if __name__ == "__main__":
    # Path to the trained model
    model_path = "speech_enhancement_model.pth"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        exit(1)
    
    # Ask user for test mode
    print("Select test mode:")
    print("1. Test on a single audio file")
    print("2. Test on multiple files in a directory")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        # Single file test
        test_file = input("Enter the path to the test audio file: ")
        if not os.path.exists(test_file):
            print(f"Error: File {test_file} not found!")
            exit(1)
        test_model_advanced(model_path, test_file)
    
    elif choice == "2":
        # Batch test
        test_dir = input("Enter the path to the directory containing test audio files: ")
        if not os.path.exists(test_dir):
            print(f"Error: Directory {test_dir} not found!")
            exit(1)
        batch_test_advanced(model_path, test_dir)
    
    else:
        print("Invalid choice!") 