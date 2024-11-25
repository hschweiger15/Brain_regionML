import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        
        static_input_size = input_size - 100
        
        # 1. Enhanced Static Feature Processing
        self.static_projection = nn.Linear(static_input_size, hidden_size)
        
        # Calculate total size of static paths
        total_static_size = sum([hidden_size // (2 ** i) for i in range(3)])
        
        # Multi-scale static processing
        self.static_paths = nn.ModuleList([
            nn.Sequential(
                nn.Linear(static_input_size, hidden_size // (2 ** i)),
                nn.BatchNorm1d(hidden_size // (2 ** i)),
                nn.GELU(),
                nn.Dropout(0.2)
            ) for i in range(3)
        ])
        
        # Add projection to match dimensions after concatenation
        self.static_fusion = nn.Linear(total_static_size, hidden_size)
        
        # 2. Advanced Waveform Processing
        self.waveform_embedder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),  # padding = kernel_size//2
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Fixed temporal convolutions with proper kernel sizes and padding
        kernel_sizes = [3, 5, 7]  # Increasing receptive field
        self.temporal_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(32, 64, 
                         kernel_size=k, 
                         padding=k//2),  # Ensures output length = input length
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.MaxPool1d(kernel_size=2, stride=2)  # Controlled downsampling
            ) for k in kernel_sizes
        ])
        
        # 3. Self-Attention Mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 4. Feature Fusion with Squeeze-and-Excitation
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),  # Ensure proper dimensions
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        
        # 5. Advanced Classifier with Skip Connections
        classifier_size = hidden_size + 64
        self.classifier_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(classifier_size if i == 0 else hidden_size,
                         hidden_size if i < 2 else output_size),
                nn.BatchNorm1d(hidden_size if i < 2 else output_size),
                nn.GELU() if i < 2 else nn.Sigmoid(),
                nn.Dropout(0.2) if i < 2 else nn.Identity()
            ) for i in range(3)
        ])
        
        # 6. Layer Normalization for feature standardization
        self.static_norm = nn.LayerNorm(hidden_size)
        self.wave_norm = nn.LayerNorm(64)
        
    def forward(self, x):
        # Split input
        static_features = x[:, :5]
        waveform = x[:, 5:].unsqueeze(1)  # [batch, 1, sequence_length]
        
        # 1. Process static features through multiple pathways
        static_projected = self.static_projection(static_features)
        static_multi = [path(static_features) for path in self.static_paths]
        static_concat = torch.cat([F.adaptive_avg_pool1d(feat.unsqueeze(-1), 1).squeeze(-1) 
                                 for feat in static_multi], dim=1)
        
        # Project concatenated features to match dimensions
        static_fused = self.static_fusion(static_concat)
        static_out = self.static_norm(static_fused + static_projected)
        
        # 2. Process waveform with multi-scale convolutions
        wave_embedded = self.waveform_embedder(waveform)  # [batch, 32, sequence_length]
        
        # Apply temporal blocks
        temporal_features = []
        for block in self.temporal_blocks:
            temp_feat = block(wave_embedded)
            temporal_features.append(temp_feat)
        
        # Ensure all features have the same size using adaptive pooling
        temporal_pooled = [F.adaptive_avg_pool1d(feat, temporal_features[-1].size(-1)) 
                          for feat in temporal_features]
        
        # Combine temporal features
        wave_concat = torch.cat(temporal_pooled, dim=1)  # Concatenate along channel dimension
        
        # 3. Apply self-attention to temporal features
        wave_attention = temporal_features[0].transpose(1, 2)  # [batch, sequence_length, channels]
        attended_features, _ = self.attention(wave_attention, wave_attention, wave_attention)
        attended_features = attended_features.transpose(1, 2)  # [batch, channels, sequence_length]
        
        # 4. Apply Squeeze-and-Excitation
        se_weights = self.se_block(attended_features)
        se_weights = se_weights.unsqueeze(-1)  # Add spatial dimension back
        wave_out = attended_features * se_weights
        wave_out = self.wave_norm(wave_out.mean(dim=2))  # Global pooling and normalization
        
        # 5. Combine features with residual connections
        combined = torch.cat([static_out, wave_out], dim=1)
        
        # 6. Final classification with skip connections
        out = combined
        intermediate_outputs = []
        for block in self.classifier_blocks[:-1]:
            block_out = block(out)
            intermediate_outputs.append(block_out)
            out = block_out + (out if out.size() == block_out.size() else out[:, :block_out.size(1)])
        
        # Final classification layer
        return self.classifier_blocks[-1](out)

class EnhancedTrainingManager:
    def __init__(self, model, criterion, optimizer, scheduler=None):
        self.device = get_device()
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Add mixup augmentation
        self.mixup_alpha = 0.2
        
        # Initialize history dictionary
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        # For M1/M2 Macs, we'll use a different approach instead of CUDA AMP
        self.use_amp = self.device == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
    def mixup_data(self, x, y):
        """Performs mixup on the input and target."""
        if np.random.random() < 0.8:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            batch_size = x.size()[0]
            index = torch.randperm(batch_size).to(self.device)
            mixed_x = lam * x + (1 - lam) * x[index, :]
            mixed_y = lam * y + (1 - lam) * y[index]
            return mixed_x, mixed_y
        return x, y
    
    def validate(self, val_loader):
        """Validation step."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return total_loss / len(val_loader), correct / total
    
    def train_epoch(self, train_loader, accumulation_steps):
        """Single epoch training step."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc='Training', leave=False)
        for i, (batch_X, batch_y) in enumerate(progress_bar):
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            # Apply mixup augmentation
            batch_X, batch_y = self.mixup_data(batch_X, batch_y)
            
            # Forward pass with or without AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y) / accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y) / accumulation_steps
                loss.backward()
            
            # Gradient accumulation step
            if (i + 1) % accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'acc': f'{(correct/total)*100:.2f}%'
            })
        
        return total_loss / len(train_loader), correct / total
    
    def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=10, accumulation_steps=4):
        """Full training loop with validation and early stopping."""
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
            
            # Print epoch results
            print(f'\nEpoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%')
            print(f'Learning Rate: {current_lr:.6f}')
            print(f'Time: {epoch_time:.2f}s')
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
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
    
    trainer = EnhancedTrainingManager(model, criterion, optimizer, scheduler)
    
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
    save_path = './training_plots'
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
