import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm, trange

from data import load_cifar10, visualize_data
from model import CIFAR10CNN, save_model

def train_model(epochs=20, batch_size=64, learning_rate=0.001):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("CIFAR-10 veri seti yükleniyor...")
    trainloader, testloader, classes = load_cifar10(batch_size=batch_size)
    
    # Visualize some training images
    print("Örnek eğitim görüntüleri görselleştiriliyor...")
    visualize_data(trainloader, classes)
    
    # Initialize the model
    print("Model başlatılıyor...")
    model = CIFAR10CNN().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Track metrics
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    # Training loop
    print("\nEğitim başlıyor...")
    print(f"Toplam epoch: {epochs}, Batch size: {batch_size}")
    start_time = time.time()
    
    # Create progress bar for epochs
    epoch_bar = trange(epochs, desc="Eğitim İlerlemesi", leave=True)
    
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        
        # Create progress bar for batches in this epoch
        batch_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for i, data in enumerate(batch_bar):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update batch progress bar
            batch_bar.set_postfix(loss=loss.item())
        
        # Calculate average training loss for this epoch
        epoch_train_loss = running_loss / len(trainloader)
        train_losses.append(epoch_train_loss)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        # Create progress bar for validation
        with torch.no_grad():
            val_bar = tqdm(testloader, desc="Doğrulama", leave=False)
            
            for data in val_bar:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update validation progress bar
                val_bar.set_postfix(loss=loss.item())
        
        # Calculate average test loss and accuracy for this epoch
        epoch_test_loss = test_loss / len(testloader)
        test_losses.append(epoch_test_loss)
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        # Update learning rate
        scheduler.step(epoch_test_loss)
        
        # Update epoch progress bar
        epoch_bar.set_postfix(train_loss=f"{epoch_train_loss:.4f}", 
                             test_loss=f"{epoch_test_loss:.4f}", 
                             accuracy=f"{accuracy:.2f}%",
                             lr=f"{optimizer.param_groups[0]['lr']:.6f}")
        
        # Also print a summary for this epoch
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Test Loss: {epoch_test_loss:.4f}, "
              f"Accuracy: {accuracy:.2f}%, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\nEğitim tamamlandı! Toplam süre: {total_time/60:.2f} dakika")
    
    # Save the trained model
    save_model(model)
    print("Model 'cifar10_model.pth' olarak kaydedildi.")
    
    # Plot training and validation loss
    print("Eğitim metrikleri grafiği oluşturuluyor...")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Eğitim Kaybı')
    plt.plot(test_losses, label='Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    plt.title('Eğitim ve Doğrulama Kaybı')
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk (%)')
    plt.title('Doğrulama Doğruluğu')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    
    # Calculate final metrics on test set
    print("\nFinak model performansı değerlendiriliyor...")
    evaluate_model(model, testloader, classes, device)
    
    return model, classes

def evaluate_model(model, testloader, classes, device):
    model.eval()
    
    # Lists to store true and predicted labels
    y_true = []
    y_pred = []
    
    # Get all predictions
    print("Test veri seti üzerinde tahminler yapılıyor...")
    with torch.no_grad():
        test_bar = tqdm(testloader, desc="Değerlendirme", leave=True)
        for data in test_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print("\nFinal Model Değerlendirme Metrikleri:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Create confusion matrix
    print("Karmaşıklık matrisi oluşturuluyor...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Tahmin Edilen Etiketler')
    plt.ylabel('Gerçek Etiketler')
    plt.title('Karmaşıklık Matrisi')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("'confusion_matrix.png' olarak kaydedildi.")

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    
    # Print welcome message
    print("=" * 50)
    print("CIFAR-10 Görüntü Sınıflandırma Modeli Eğitimi")
    print("=" * 50)
    
    # Train the model
    train_model() 