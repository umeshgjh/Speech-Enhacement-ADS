import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

# Step 1: Load and Process ODAQ Dataset
def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

# Step 2: Feature Extraction (MFCCs & Spectrograms)
def extract_features(audio, sr=16000, max_frames=400):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    # Pad or truncate to a fixed number of frames
    if mfccs.shape[1] < max_frames:
        # Pad with zeros if shorter than max_frames
        pad_width = ((0, 0), (0, max_frames - mfccs.shape[1]))
        mfccs = np.pad(mfccs, pad_width, mode='constant')
    elif mfccs.shape[1] > max_frames:
        # Truncate if longer than max_frames
        mfccs = mfccs[:, :max_frames]
    
    return mfccs

# Step 3: Define a PyTorch Dataset for Training
class AudioDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_frames=400):
        self.data_dir = data_dir
        self.file_list = []
        # Recursively find all .wav files in the data directory and its subdirectories
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".wav"):
                    self.file_list.append(os.path.join(root, file))
        self.transform = transform
        self.max_frames = max_frames
        print(f"Found {len(self.file_list)} .wav files in {data_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        audio = load_audio(file_path)
        features = extract_features(audio, max_frames=self.max_frames)
        if self.transform:
            features = self.transform(features)
        return torch.tensor(features, dtype=torch.float32)

# Step 4: Create CNN Model for Speech Enhancement
import torch.nn as nn
import torch.optim as optim

class SpeechEnhancementCNN(nn.Module):
    def __init__(self, input_channels=1, n_mfcc=40, max_frames=400):
        super(SpeechEnhancementCNN, self).__init__()
        
        # Calculate the output size after convolutions
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size after pooling
        # After two pooling layers, the size will be reduced by a factor of 4
        # So the output size will be (n_mfcc/4) * (max_frames/4)
        conv_output_size = (n_mfcc // 4) * (max_frames // 4) * 64
        
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_mfcc * max_frames)  # Output size matches input size
        
    def forward(self, x):
        # Apply convolutions and pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Reshape to match input dimensions
        batch_size = x.size(0)
        x = x.view(batch_size, 40, 400)  # Reshape to (batch_size, n_mfcc, max_frames)
        
        return x

# Step 5: Training the Model
def train_model(model, dataset, epochs=10, batch_size=32, learning_rate=0.001):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, data in enumerate(dataloader):
            # Add channel dimension for CNN
            inputs = data.unsqueeze(1)  # Shape: [batch_size, 1, n_mfcc, max_frames]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss (comparing output with input for autoencoder-like behavior)
            loss = criterion(outputs, data)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print batch progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        # Print epoch summary
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_epoch_loss:.4f}")
    
    print("Training Complete!")

# Step 6: Deployment on Jetson Nano (Real-Time Processing)
def deploy_model(model, input_audio):
    model.eval()
    input_features = extract_features(input_audio)
    with torch.no_grad():
        output = model(torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    return output

# Running the full pipeline
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define parameters
    data_dir = "/media/umeshgjh/New Volume/Study/ADS PROJECT/ODAQ"
    max_frames = 400
    n_mfcc = 40
    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    
    # Create dataset and model
    print("Loading dataset...")
    dataset = AudioDataset(data_dir, max_frames=max_frames)
    
    print("Creating model...")
    model = SpeechEnhancementCNN(input_channels=1, n_mfcc=n_mfcc, max_frames=max_frames)
    
    # Print model summary
    print(f"Model architecture:\n{model}")
    
    # Train the model
    print("Starting training...")
    train_model(model, dataset, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    
    # Save the trained model
    print("Saving model...")
    torch.save(model.state_dict(), "speech_enhancement_model.pth")
    print("Model saved successfully!")

