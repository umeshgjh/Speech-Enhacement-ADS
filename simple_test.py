import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import librosa
import librosa.display
import soundfile as sf
from train import SpeechEnhancementCNN, load_audio, extract_features

def simple_test(model_path, test_audio_path, output_dir="simple_test_results"):
    """
    Simple test of the speech enhancement model without complex audio reconstruction.
    
    Args:
        model_path: Path to the saved model
        test_audio_path: Path to the test audio file
        output_dir: Directory to save results
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
    
    # Save the original audio
    original_path = os.path.join(output_dir, "original_audio.wav")
    sf.write(original_path, audio, sr)
    print(f"Original audio saved to {original_path}")
    
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
    plot_path = os.path.join(output_dir, "enhancement_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Visualization saved to {plot_path}")
    
    # Calculate and save statistics
    stats = {
        "Mean difference": np.mean(diff),
        "Max difference": np.max(np.abs(diff)),
        "Standard deviation of difference": np.std(diff),
        "Correlation between original and enhanced": np.corrcoef(features.flatten(), enhanced_features.flatten())[0, 1]
    }
    
    # Save statistics to file
    stats_path = os.path.join(output_dir, "statistics.txt")
    with open(stats_path, "w") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value:.4f}\n")
    
    print(f"Statistics saved to {stats_path}")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    
    return enhanced_features

def batch_simple_test(model_path, test_dir, output_dir="simple_test_results"):
    """
    Run simple tests on multiple audio files in a directory.
    
    Args:
        model_path: Path to the saved model
        test_dir: Directory containing test audio files
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all wav files in the test directory
    test_files = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith((".wav", ".mp3", ".webm")):
                test_files.append(os.path.join(root, file))
    
    print(f"Found {len(test_files)} test files")
    
    # Process each file
    for i, test_file in enumerate(test_files):
        print(f"\nProcessing file {i+1}/{len(test_files)}: {test_file}")
        
        # Create a subdirectory for this file
        file_name = os.path.basename(test_file).split('.')[0]
        file_output_dir = os.path.join(output_dir, file_name)
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Test the model on this file
        simple_test(model_path, test_file, file_output_dir)

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
        simple_test(model_path, test_file)
    
    elif choice == "2":
        # Batch test
        test_dir = input("Enter the path to the directory containing test audio files: ")
        if not os.path.exists(test_dir):
            print(f"Error: Directory {test_dir} not found!")
            exit(1)
        batch_simple_test(model_path, test_dir)
    
    else:
        print("Invalid choice!") 