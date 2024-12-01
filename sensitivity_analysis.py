import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from cnn import TrafficLightCNN
from PIL import Image
model = TrafficLightCNN()
model.load_state_dict(torch.load('./models/cnn/best_cnn.pth'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = ImageFolder(root='./data/cropped traffic lights', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)
def test_brightness_factor(factor):
    brightness_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ColorJitter(brightness=(factor, factor)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    brightness_dataset = ImageFolder(root='./data/cropped traffic lights', transform=brightness_transform)
    brightness_loader = DataLoader(brightness_dataset, batch_size=32, shuffle=False)
    return evaluate_model(brightness_loader)
def test_contrast_factor(factor):
    contrast_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ColorJitter(contrast=(factor, factor)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    contrast_dataset = ImageFolder(root='./data/cropped traffic lights', transform=contrast_transform)
    contrast_loader = DataLoader(contrast_dataset, batch_size=32, shuffle=False)
    return evaluate_model(contrast_loader)
def add_noise(image, noise_factor=0.2):
    np_image = np.array(image) / 255.0
    noise = np.random.randn(*np_image.shape) * noise_factor
    np_image = np.clip(np_image + noise, 0, 1)
    np_image = (np_image * 255).astype(np.uint8)
    return Image.fromarray(np_image)
def test_noise_factor(factor):
    noise_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Lambda(lambda img: add_noise(img, noise_factor=factor)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    noise_dataset = ImageFolder(root='./data/cropped traffic lights', transform=noise_transform)
    noise_loader = DataLoader(noise_dataset, batch_size=32, shuffle=False)
    return evaluate_model(noise_loader)
def evaluate_model(loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = 1 - predicted
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

brightness_factors = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
contrast_factors = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
noise_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

print("Testing Brightness:")
for factor in brightness_factors:
    print(f"Brightness Factor: {factor}")
    test_brightness_factor(factor)

print("\nTesting Contrast:")
for factor in contrast_factors:
    print(f"Contrast Factor: {factor}")
    test_contrast_factor(factor)

print("\nTesting Noise:")
for factor in noise_factors:
    print(f"Noise Factor: {factor}")
    test_noise_factor(factor)
