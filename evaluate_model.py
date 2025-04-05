import os
import numpy as np
import torch
import librosa
import soundfile as sf
from scipy.stats import pearsonr
from train import SpeechEnhancementCNN, load_audio, extract_features
from test_model_advanced import mfcc_to_audio

def calculate_snr(original, enhanced):
    """
    Calculate Signal-to-Noise Ratio (SNR).
    
    Args:
        original: Original signal
        enhanced: Enhanced signal
    
    Returns:
        SNR in dB
    """
    # Ensure both signals have the same length
    min_len = min(len(original), len(enhanced))
    original = original[:min_len]
    enhanced = enhanced[:min_len]
    
    # Calculate noise (difference between original and enhanced)
    noise = original - enhanced
    
    # Calculate signal and noise power
    signal_power = np.sum(original ** 2) / len(original)
    noise_power = np.sum(noise ** 2) / len(noise)
    
    # Calculate SNR
    if noise_power == 0:
        return float('inf')  # No noise
    else:
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

def calculate_pesq(original_path, enhanced_path):
    """
    Calculate PESQ (Perceptual Evaluation of Speech Quality).
    Note: This requires the pesq package to be installed.
    
    Args:
        original_path: Path to original audio file
        enhanced_path: Path to enhanced audio file
    
    Returns:
        PESQ score
    """
    try:
        from pesq import pesq
        
        # Load audio files
        original, sr_orig = sf.read(original_path)
        enhanced, sr_enh = sf.read(enhanced_path)
        
        # Ensure both signals have the same sample rate
        if sr_orig != sr_enh:
            print(f"Warning: Sample rates differ ({sr_orig} vs {sr_enh}). Using {sr_orig}.")
        
        # Calculate PESQ
        score = pesq(sr_orig, original, enhanced, 'wb')
        return score
    
    except ImportError:
        print("PESQ package not installed. Skipping PESQ calculation.")
        print("To install: pip install pesq")
        return None
    except Exception as e:
        print(f"Error calculating PESQ: {e}")
        return None

def calculate_stoi(original, enhanced, sr=16000):
    """
    Calculate STOI (Short-Time Objective Intelligibility).
    Note: This requires the pystoi package to be installed.
    
    Args:
        original: Original signal
        enhanced: Enhanced signal
        sr: Sample rate
    
    Returns:
        STOI score
    """
    try:
        from pystoi import stoi
        
        # Ensure both signals have the same length
        min_len = min(len(original), len(enhanced))
        original = original[:min_len]
        enhanced = enhanced[:min_len]
        
        # Calculate STOI
        score = stoi(original, enhanced, sr, extended=False)
        return score
    
    except ImportError:
        print("pystoi package not installed. Skipping STOI calculation.")
        print("To install: pip install pystoi")
        return None
    except Exception as e:
        print(f"Error calculating STOI: {e}")
        return None

def evaluate_model(model_path, test_audio_path, output_dir="evaluation_results"):
    """
    Evaluate the trained speech enhancement model on a single audio file.
    
    Args:
        model_path: Path to the saved model
        test_audio_path: Path to the test audio file
        output_dir: Directory to save evaluation results
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
    
    # Save the original audio
    original_path = os.path.join(output_dir, "original_audio.wav")
    sf.write(original_path, audio, sr)
    
    # Reconstruct audio from MFCCs
    try:
        print("Reconstructing audio from original MFCCs...")
        original_reconstructed = mfcc_to_audio(features, sr=sr)
        original_reconstructed_path = os.path.join(output_dir, "original_reconstructed.wav")
        sf.write(original_reconstructed_path, original_reconstructed, sr)
        
        print("Reconstructing audio from enhanced MFCCs...")
        enhanced_audio = mfcc_to_audio(enhanced_features, sr=sr)
        enhanced_path = os.path.join(output_dir, "enhanced_audio.wav")
        sf.write(enhanced_path, enhanced_audio, sr)
        
        # Calculate objective metrics
        print("\nCalculating objective metrics...")
        
        # 1. SNR
        snr = calculate_snr(original_reconstructed, enhanced_audio)
        print(f"SNR: {snr:.2f} dB")
        
        # 2. PESQ
        pesq_score = calculate_pesq(original_reconstructed_path, enhanced_path)
        if pesq_score is not None:
            print(f"PESQ: {pesq_score:.2f}")
        
        # 3. STOI
        stoi_score = calculate_stoi(original_reconstructed, enhanced_audio, sr=sr)
        if stoi_score is not None:
            print(f"STOI: {stoi_score:.2f}")
        
        # 4. Correlation between original and enhanced MFCCs
        correlation, _ = pearsonr(features.flatten(), enhanced_features.flatten())
        print(f"MFCC Correlation: {correlation:.4f}")
        
        # Save metrics to file
        metrics_path = os.path.join(output_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"SNR: {snr:.2f} dB\n")
            if pesq_score is not None:
                f.write(f"PESQ: {pesq_score:.2f}\n")
            if stoi_score is not None:
                f.write(f"STOI: {stoi_score:.2f}\n")
            f.write(f"MFCC Correlation: {correlation:.4f}\n")
        
        print(f"Metrics saved to {metrics_path}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Note: Audio reconstruction from MFCCs is challenging and may not produce high-quality results.")

def batch_evaluate(model_path, test_dir, output_dir="evaluation_results"):
    """
    Evaluate the model on multiple audio files in a directory.
    
    Args:
        model_path: Path to the saved model
        test_dir: Directory containing test audio files
        output_dir: Directory to save evaluation results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all wav files in the test directory
    test_files = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith(".wav"):
                test_files.append(os.path.join(root, file))
    
    print(f"Found {len(test_files)} test files")
    
    # Process each file
    all_metrics = []
    for i, test_file in enumerate(test_files):
        print(f"\nProcessing file {i+1}/{len(test_files)}: {test_file}")
        
        # Create a subdirectory for this file
        file_name = os.path.basename(test_file).replace(".wav", "")
        file_output_dir = os.path.join(output_dir, file_name)
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Evaluate the model on this file
        evaluate_model(model_path, test_file, file_output_dir)
        
        # Collect metrics
        metrics_path = os.path.join(file_output_dir, "metrics.txt")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = f.readlines()
            all_metrics.append((file_name, metrics))
    
    # Calculate and save average metrics
    if all_metrics:
        print("\nCalculating average metrics across all files...")
        
        # Initialize counters
        snr_sum = 0
        pesq_sum = 0
        stoi_sum = 0
        correlation_sum = 0
        snr_count = 0
        pesq_count = 0
        stoi_count = 0
        correlation_count = 0
        
        # Sum up metrics
        for _, metrics in all_metrics:
            for line in metrics:
                if line.startswith("SNR:"):
                    snr = float(line.split(":")[1].strip().split()[0])
                    snr_sum += snr
                    snr_count += 1
                elif line.startswith("PESQ:"):
                    pesq = float(line.split(":")[1].strip())
                    pesq_sum += pesq
                    pesq_count += 1
                elif line.startswith("STOI:"):
                    stoi = float(line.split(":")[1].strip())
                    stoi_sum += stoi
                    stoi_count += 1
                elif line.startswith("MFCC Correlation:"):
                    correlation = float(line.split(":")[1].strip())
                    correlation_sum += correlation
                    correlation_count += 1
        
        # Calculate averages
        avg_metrics = []
        if snr_count > 0:
            avg_snr = snr_sum / snr_count
            avg_metrics.append(f"Average SNR: {avg_snr:.2f} dB")
        
        if pesq_count > 0:
            avg_pesq = pesq_sum / pesq_count
            avg_metrics.append(f"Average PESQ: {avg_pesq:.2f}")
        
        if stoi_count > 0:
            avg_stoi = stoi_sum / stoi_count
            avg_metrics.append(f"Average STOI: {avg_stoi:.2f}")
        
        if correlation_count > 0:
            avg_correlation = correlation_sum / correlation_count
            avg_metrics.append(f"Average MFCC Correlation: {avg_correlation:.4f}")
        
        # Save average metrics
        avg_metrics_path = os.path.join(output_dir, "average_metrics.txt")
        with open(avg_metrics_path, "w") as f:
            f.write("\n".join(avg_metrics))
        
        print(f"Average metrics saved to {avg_metrics_path}")
        for metric in avg_metrics:
            print(metric)

if __name__ == "__main__":
    # Path to the trained model
    model_path = "speech_enhancement_model.pth"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        exit(1)
    
    # Ask user for evaluation mode
    print("Select evaluation mode:")
    print("1. Evaluate on a single audio file")
    print("2. Evaluate on multiple files in a directory")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        # Single file evaluation
        test_file = input("Enter the path to the test audio file: ")
        if not os.path.exists(test_file):
            print(f"Error: File {test_file} not found!")
            exit(1)
        evaluate_model(model_path, test_file)
    
    elif choice == "2":
        # Batch evaluation
        test_dir = input("Enter the path to the directory containing test audio files: ")
        if not os.path.exists(test_dir):
            print(f"Error: Directory {test_dir} not found!")
            exit(1)
        batch_evaluate(model_path, test_dir)
    
    else:
        print("Invalid choice!") 