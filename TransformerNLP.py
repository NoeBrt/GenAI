import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Load IMDb dataset using Keras (I didn't succeed in doing it with torch)
def load_imdb_dataset(num_words=10000):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)
    return x_train, y_train, x_test, y_test

# Initialize tokenizer
def initialize_tokenizer(model_name="bert-base-uncased"):
    return BertTokenizer.from_pretrained(model_name)

# Tokenize and pad the input data. The text is split by space and tokenized.
def preprocess_data(x_data, tokenizer, max_length=256):
    # Load the word index mapping from Keras
    word_index = tf.keras.datasets.imdb.get_word_index()
    
    # Create a reverse mapping, noting that the indices are offset by 3
    index_to_word = {i+3: word for word, i in word_index.items()}
    index_to_word[0] = "[PAD]"
    index_to_word[1] = "[START]"
    index_to_word[2] = "[UNK]"
    index_to_word[3] = "[UNUSED]"
    
    # Decode each review from integers to words
    texts = [" ".join([index_to_word.get(i, "[UNK]") for i in review]) for review in x_data]
    
    # Tokenize the decoded text
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return encodings


# Convert the input data into a PyTorch Dataset and DataLoader.
def create_dataloader(encodings, labels, batch_size, shuffle=True):
    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], torch.tensor(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

#build the model
def build_model(model_name="bert-base-uncased", num_labels=2):
    return BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

#train the model
def train_model(model, train_loader, val_loader, epochs, lr, device):
    optimizer = AdamW(model.parameters(), lr=lr)
    model.to(device)
    train_losses = []
    val_accuracies = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        model.eval()
        correct = 0
        total = 0
        #stop the gradient calculation to compute the validation accuracy
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
    return train_losses, val_accuracies

#evaluate the model 
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            #get the predicted class with the highest probability
            preds = torch.argmax(outputs.logits, dim=1)
            #append the predictions and labels to the list
            all_preds.extend(preds.cpu().numpy())
            #append the labels to the list
            all_labels.extend(labels.cpu().numpy())
    #compute the test accuracy
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
    plt.legend()
    plt.title("Training Loss & Validation Accuracy")
    plt.show()
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train, x_test, y_test = load_imdb_dataset(num_words=10000)
    tokenizer = initialize_tokenizer()
    train_encodings = preprocess_data(x_train, tokenizer, max_length=args.max_length)
    test_encodings = preprocess_data(x_test, tokenizer, max_length=args.max_length)
    # Create DataLoader a mix of a dataset and a Sampler for torch
    train_loader = create_dataloader(train_encodings, y_train, args.batch_size, shuffle=True)
    full_test_dataset = TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"], torch.tensor(y_test))
    val_size = len(full_test_dataset) // 2
    test_size = len(full_test_dataset) - val_size
    val_dataset, test_dataset = random_split(full_test_dataset, [val_size, test_size])
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = build_model()
    train_losses, val_accuracies = train_model(model, train_loader, val_loader, args.epochs, args.lr, device)
    save_model(model, args.save_path)
    test_accuracy, pred_labels, true_labels = evaluate_model(model, test_loader, device)
    print("Test Accuracy:", test_accuracy)
    visualize_results(train_losses, val_accuracies, true_labels, pred_labels)

def inference(input_text):
    model = BertForSequenceClassification.from_pretrained("./saved_bert_imdb")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print(input_text)
    encoding = tokenizer(input_text, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    class_names = ["Negative", "Positive"]
    print(f"The text '{input_text}' is overall {class_names[predicted_class]} with probability {probabilities[0][predicted_class].item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT on IMDb sentiment classification using PyTorch")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length for tokenizer")
    parser.add_argument("--save_path", type=str, default="./saved_bert_imdb", help="Directory to save the model")
    parser.add_argument("--inference", action="store_true", help="Perform inference on a sample text")
    parser.add_argument("--input_text", type=str, default="This movie was great!", help="Input text for inference")
    args = parser.parse_args()
    if args.inference:
        inference(args.input_text)
    else:
        main(args)
