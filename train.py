import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from cnn import EnhancedTrafficLightCNN
import numpy as np
from sklearn.metrics import classification_report # type: ignore
import os

def train_model():
    # Enhanced data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Larger input size
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_data = datasets.ImageFolder('cropped_dataset/train', transform=train_transform)
    val_data = datasets.ImageFolder('cropped_dataset/val', transform=val_transform)

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Classes: {train_data.classes}")

    # Handle class imbalance
    class_counts = [len([x for x in train_data.samples if x[1] == i]) for i in range(len(train_data.classes))]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[i] for _, i in train_data.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_data, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    # Initialize enhanced model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedTrafficLightCNN(num_classes=len(train_data.classes)).to(device)
    
    # Enhanced optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    best_accuracy = 0.0
    print("Starting enhanced training...")

    for epoch in range(25):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        scheduler.step(val_accuracy)

        print(f'Epoch {epoch+1}/25 | Loss: {running_loss/len(train_loader):.4f} | Acc: {val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': train_data.classes,
                'accuracy': best_accuracy
            }, 'best_enhanced_model.pth')
            print(f'✓ New best model saved: {best_accuracy:.4f}')

    # Save final model
    torch.save(model.state_dict(), 'final_enhanced_model.pth')
    
    # Generate report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_data.classes))
    
    return best_accuracy

if __name__ == '__main__':
    best_acc = train_model()
    print(f"\nBest validation accuracy: {best_acc:.4f}")