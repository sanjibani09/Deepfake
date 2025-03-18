Deepfake Detection
Deepfake Detection Using 3D ResNet
Overview

This project implements a deepfake detection system using a 3D ResNet (R3D-18) model from PyTorch's torchvision.models.video module. The model is trained on real and fake videos to classify whether a given video is genuine or manipulated.
Features

    Uses 3D ResNet (R3D-18) for deepfake classification.
    Loads and processes videos using OpenCV and PyTorch.
    Implements incremental training with checkpoints to avoid loss of progress.
    Real-time progress tracking using tqdm.
    Uses binary classification with BCEWithLogitsLoss.

Dataset Structure

The dataset consists of real and fake videos stored in structured directories:

D:\Dataset\
    ├── Celeb-real\   # Real videos
    ├── Youtube-real\ # Real videos
    ├── Celeb-synthesis\  # Fake videos

Prerequisites

Ensure you have the following installed:

    Python 3.8+
    PyTorch
    Torchvision
    OpenCV
    NumPy
    TQDM
    Gradio (optional for real-time testing)

You can install dependencies using:

pip install torch torchvision opencv-python numpy tqdm gradio

Code Structure

    VideoDataset: Custom PyTorch dataset class for loading and processing videos.
    DeepfakeDetector: Neural network model using 3D ResNet for classification.
    train(): Training function with checkpointing and progress tracking.
    train_loader, val_loader: PyTorch data loaders for training and validation.
    process_video(): Extracts frames, resizes them, and normalizes video input.

Training the Model

Run the script to train the deepfake detection model:

python train.py

Training Settings

    Epochs: 10 (configurable)
    Learning Rate: 0.0001
    Batch Size: 4
    Training-Validation Split: 80%-20%
    Device: Uses GPU if available, otherwise CPU

Checkpointing

The training script automatically saves checkpoints every 10 processed videos in checkpoint.pth. It contains:

    Model state dictionary
    Optimizer state dictionary
    Epoch number
    Processed videos count

If training is interrupted, it resumes from the last checkpoint.
Model Inference

To test a video using a trained model, use:

model.eval()
video_tensor = dataset.process_video("path/to/video.mp4")
video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension
output = model(video_tensor)
prediction = torch.sigmoid(output).item()
print("Fake" if prediction > 0.5 else "Real")

Future Improvements

    Improve data augmentation techniques.
    Increase dataset diversity.
    Optimize model performance using different architectures.
