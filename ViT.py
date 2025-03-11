import argparse
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.optim import AdamW
from PIL import Image
# I use torch because i had a lot of trouble with tensorflow


# Define the labels for CIFAR-10
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define the function to get the transform for the image
def get_transform(feature_extractor):
    def transform(image):
        inputs = feature_extractor(image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)
    return transform

def load_dataset(feature_extractor, data_dir="./data"):
    transform = get_transform(feature_extractor)
    train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def build_model(model_name="google/vit-base-patch16-224", num_labels=10):
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    return model

def train_model(model, train_loader, val_loader, epochs, lr, device):
    optimizer = AdamW(model.parameters(), lr=lr)
    model.to(device)
    train_losses = []
    val_accuracies = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images, labels=labels)
            loss = outputs.loss
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Accuracy: {val_acc:.4f}")
    return train_losses, val_accuracies

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return test_accuracy, all_preds, all_labels

def save_model(model, save_path):
    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

def visualize_results(train_losses, val_accuracies, true_labels, pred_labels):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.title("Training Loss & Validation Accuracy")
    plt.legend()
    plt.show()
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def inference(image_path, model_path="./saved_vit_cifar10"):
    model = ViTForImageClassification.from_pretrained(model_path)
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    image = Image.open(image_path).convert("RGB")
    transform = get_transform(feature_extractor)
    pixel_values = transform(image)
    pixel_values = pixel_values.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    print("Probabilities:", probabilities)
    print(f"Predicted Class: {predicted_class} - {labels[predicted_class]}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    train_dataset, test_dataset = load_dataset(feature_extractor, data_dir=args.data_dir)
    val_size = len(test_dataset) // 2
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])
    #Put the dataset into torch dataloader 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = build_model(num_labels=10)
    train_losses, val_accuracies = train_model(model, train_loader, val_loader, args.epochs, args.lr, device)
    save_model(model, args.save_path)
    test_acc, pred_labels, true_labels = evaluate_model(model, test_loader, device)
    print("Test Accuracy:", test_acc)
    visualize_results(train_losses, val_accuracies, true_labels, pred_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ViT on CIFAR-10 using PyTorch")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="./saved_vit_cifar10", help="Directory to save the model")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to download CIFAR-10 data")
    parser.add_argument("--inference", action="store_true", help="Perform inference on a sample image")
    parser.add_argument("--image_path", type=str, default=None, help="Path to the image file for inference")
    parser.add_argument("--model_path", type=str, default="./vit_model", help="Path to the saved model for inference")
    args = parser.parse_args()
    if args.inference:
        if args.image_path is None:
            print("Please provide an image path using --image_path for inference.")
        else:
            inference(args.image_path, args.model_path)
    else:
        main(args)
