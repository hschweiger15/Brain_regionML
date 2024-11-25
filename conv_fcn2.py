import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import time
import os

# Device detection for M1/M2 support
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class EnhancedModel(nn.Module):
    def __init__(self, input_size=105, hidden_size=64, output_size=1):
        super().__init__()
        
        static_input_size = input_size - 100  # 5 features
        
        # Projection layer for residual connection
        self.static_projection = nn.Linear(static_input_size, hidden_size)
        
        # Static features pathway with residual connections
        self.static_block1 = nn.Sequential(
            nn.Linear(static_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.static_block2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.static_block3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU()
        )
        
        # Waveform pathway remains the same
        self.waveform_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(2),
            
            nn.Flatten()
        )
        
        cnn_output_size = 64 * 12  # Adjusted for pooling
        
        self.waveform_fcn = nn.Sequential(
            nn.Linear(cnn_output_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size//2, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Split input
        static_features = x[:, :5]
        waveform = x[:, 5:].unsqueeze(1)
        
        # Project static features for residual connection
        static_projected = self.static_projection(static_features)
        
        # First static block with residual
        static_out1 = self.static_block1(static_features)
        static_out1 = static_out1 + static_projected  # First residual connection
        
        # Second static block with residual
        static_out2 = self.static_block2(static_out1)
        static_out2 = static_out2 + static_out1  # Second residual connection
        
        # Final static features
        static_out = self.static_block3(static_out2)
        
        # Process waveform
        waveform_cnn = self.waveform_cnn(waveform)
        waveform_out = self.waveform_fcn(waveform_cnn)
        
        # Combine features
        combined = torch.cat((static_out, waveform_out), dim=1)
        return self.classifier(combined)

class ImprovedTrainingManager:
    def __init__(self, model, criterion, optimizer, scheduler=None):
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model.to(self.device)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        self.smoothing = 0.1
    
    def criterion_with_smoothing(self, outputs, targets):
        smooth_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.criterion(outputs, smooth_targets)
    
    def train_epoch(self, train_loader, accumulation_steps):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc='Training', leave=False)
        for i, (batch_X, batch_y) in enumerate(progress_bar):
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            outputs = self.model(batch_X)
            loss = self.criterion_with_smoothing(outputs, batch_y)
            
            # Normalize loss to account for accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'acc': f'{(correct/total)*100:.2f}%'
            })
        
        return total_loss / len(train_loader), correct / total

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion_with_smoothing(outputs, batch_y)
                
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return total_loss / len(val_loader), correct / total

    def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=10, accumulation_steps=4):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, accumulation_steps)
            
            # Validation phase
            val_loss, val_acc = self.validate(val_loader)
            
            # Record time and learning rate
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(epoch_time)
            self.history['learning_rates'].append(current_lr)
            
            print(f'\nEpoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%')
            print(f'Learning Rate: {current_lr:.6f}')
            print(f'Time: {epoch_time:.2f}s')
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, 'best_model.pth')
                print("✓ Saved new best model")
            else:
                patience_counter += 1
                print(f"! Validation loss didn't improve for {patience_counter} epochs")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f'\nEarly stopping triggered after epoch {epoch+1}')
                break
        
        # Load best model
        checkpoint = torch.load('best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nTraining completed. Best validation loss: {checkpoint['val_loss']:.4f}")
        print(f"Best validation accuracy: {checkpoint['val_acc']*100:.2f}%")
        
        return self.history

def plot_training_history(history, save_path='./plots'):
    """
    Plot and save training metrics from the training history.
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Set style for better-looking plots - using a built-in style
    plt.style.use('bmh')  # Alternative built-in style that looks professional
    
    # Common plotting parameters
    plot_params = {
        'figure.figsize': (10, 6),
        'axes.grid': True,
        'lines.linewidth': 2,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    }
    plt.rcParams.update(plot_params)
    
    # Plot 1: Loss curves
    plt.figure()
    plt.plot(history['train_loss'], label='Training Loss', color='#2E86C1', alpha=0.7)
    plt.plot(history['val_loss'], label='Validation Loss', color='#E74C3C', alpha=0.7)
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Accuracy curves
    plt.figure()
    plt.plot(history['train_acc'], label='Training Accuracy', color='#2E86C1', alpha=0.7)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='#E74C3C', alpha=0.7)
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'accuracy_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Learning rate over time
    plt.figure()
    plt.plot(history['learning_rates'], color='#27AE60', alpha=0.7)
    plt.title('Learning Rate Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'learning_rate_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Training time per epoch
    plt.figure()
    plt.plot(history['epoch_times'], color='#8E44AD', alpha=0.7)
    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'epoch_times.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics to CSV for future reference
    metrics_df = pd.DataFrame({
        'epoch': range(len(history['train_loss'])),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc'],
        'learning_rate': history['learning_rates'],
        'epoch_time': history['epoch_times']
    })
    metrics_df.to_csv(os.path.join(save_path, 'training_metrics.csv'), index=False)

    
def train_model(X_train, y_train, X_test, y_test, save_path="plots", batch_size=64):
    device = get_device()
    
    # Initialize improved model and training components
    model = EnhancedModel()
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Prepare data
    train_size = int(0.8 * len(X_train))
    train_dataset = TensorDataset(X_train[:train_size], y_train[:train_size])
    val_dataset = TensorDataset(X_train[train_size:], y_train[train_size:])
    
    # For M1/M2 Macs, adjust num_workers
    import platform
    num_workers = 0 if platform.processor() == 'arm' else 4
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device != "cpu" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if device != "cpu" else False
    )
    
    trainer = ImprovedTrainingManager(model, criterion, optimizer, scheduler)
    
    accumulation_steps = 4
    effective_batch_size = batch_size * accumulation_steps
    print(f"Training with effective batch size: {effective_batch_size}")
    
    # Train the model
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=100,
        early_stopping_patience=10,
        accumulation_steps=accumulation_steps
    )
    
    # Plot and save training history
    plot_training_history(history, save_path)
    
    return history

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Create save directory for plots
    save_path = './training_plots2'
    os.makedirs(save_path, exist_ok=True)
    
    # Load and prepare data
    train_data = pd.read_csv('./training_data.csv')
    test_data = pd.read_csv('./test_data.csv')
    
    X_train = torch.FloatTensor(train_data.drop('label', axis=1).values)
    y_train = torch.FloatTensor(train_data['label'].values).reshape(-1, 1)
    X_test = torch.FloatTensor(test_data.drop('label', axis=1).values)
    y_test = torch.FloatTensor(test_data['label'].values).reshape(-1, 1)
    
    # Train model and save plots
    history = train_model(
        X_train, y_train,
        X_test, y_test,
        save_path=save_path,
        batch_size=64
    )
    
    # Save final metrics to a file
    with open(os.path.join(save_path, 'final_metrics.txt'), 'w') as f:
        f.write(f"Final Training Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Validation Loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"Final Training Accuracy: {history['train_acc'][-1]*100:.2f}%\n")
        f.write(f"Final Validation Accuracy: {history['val_acc'][-1]*100:.2f}%\n")
        f.write(f"Total Training Time: {sum(history['epoch_times']):.2f} seconds\n")

if __name__ == "__main__":
    main()
