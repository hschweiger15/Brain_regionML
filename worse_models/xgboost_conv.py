import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import shap
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

import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import shap
import time
import os

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class CNNFeatureExtractor(nn.Module):
    """CNN model for supervised feature extraction from waveform data"""
    def __init__(self, hidden_size=64):
        super().__init__()
        
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
        
        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(cnn_output_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU()
        )
        
        # Classification head for pretraining
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_features=False):
        waveform = x.unsqueeze(1)  # Add channel dimension
        cnn_features = self.waveform_cnn(waveform)
        features = self.feature_layers(cnn_features)
        
        if return_features:
            return features
            
        return self.classifier(features)

class HybridModel:
    """Combines CNN Feature Extractor with XGBoost"""
    def __init__(self, cnn_hidden_size=64):
        self.device = get_device()
        self.cnn = CNNFeatureExtractor(hidden_size=cnn_hidden_size).to(self.device)
        self.xgb_model = None
        self.feature_names = None
        
    def extract_features(self, X):
        """Extract features using CNN"""
        self.cnn.eval()
        waveform_data = X[:, 5:]  # Assumes first 5 features are static
        static_features = X[:, :5]
        
        with torch.no_grad():
            waveform_features = self.cnn(waveform_data.to(self.device), return_features=True)
            
        # Combine static features with CNN features
        combined_features = np.hstack([
            static_features.cpu().numpy(),
            waveform_features.cpu().numpy()
        ])
        
        return combined_features
    
    def train_cnn(self, train_loader, val_loader, epochs=50):
        """Train CNN feature extractor using supervised learning"""
        print(f"Training CNN on device: {self.device}")
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(self.cnn.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.cnn.train()
            train_loss = 0
            correct = 0
            total = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_X, batch_y in progress_bar:
                waveform = batch_X[:, 5:].to(self.device)  # Only waveform data
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.cnn(waveform)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct/total:.2f}%'
                })
            
            train_loss = train_loss / len(train_loader)
            train_acc = 100 * correct / total
            
            # Validation
            self.cnn.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    waveform = batch_X[:, 5:].to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.cnn(waveform)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            print(f'\nEpoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.cnn.state_dict(), 'best_cnn_extractor.pth')
                print("✓ Saved new best model")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        
        # Load best model
        self.cnn.load_state_dict(torch.load('best_cnn_extractor.pth'))
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model on combined features"""
        print("Extracting features for training...")
        train_features = self.extract_features(X_train)
        
        # Create feature names
        static_names = [f'static_{i}' for i in range(5)]
        cnn_names = [f'cnn_{i}' for i in range(train_features.shape[1]-5)]
        self.feature_names = static_names + cnn_names
        
        # Prepare validation set
        eval_set = None
        if X_val is not None and y_val is not None:
            val_features = self.extract_features(X_val)
            eval_set = [(val_features, y_val.numpy())]
        
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist'
        }
        
        print("Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(**params)
        self.xgb_model.fit(
            train_features, y_train.numpy(),
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=True
        )
        
        return self.xgb_model
    
    def predict(self, X):
        """Make predictions using the hybrid model"""
        features = self.extract_features(X)
        return self.xgb_model.predict_proba(features)[:, 1]

class ModelAnalyzer:
    """Analyze and visualize model performance and features"""
    def __init__(self, model, feature_names=None, save_dir='./xgboost_conv_plots'):
        self.model = model
        self.feature_names = feature_names
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
    def save_plot(self, plt, name):
        """Helper function to save plots"""
        path = os.path.join(self.save_dir, f"{name}.png")
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved plot to {path}")
        
    def plot_feature_importance(self, importance_type='weight'):
        """Plot XGBoost feature importance"""
        importance = self.model.xgb_model.get_booster().get_score(
            importance_type=importance_type
        )
        
        plt.figure(figsize=(12, 8))
        sorted_idx = np.argsort(list(importance.values()))
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        plt.barh(pos, [importance[k] for k in sorted(importance.keys())])
        plt.yticks(pos, [list(importance.keys())[idx] for idx in sorted_idx])
        plt.xlabel(f'Feature Importance ({importance_type})')
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        
        self.save_plot(plt, f'feature_importance_{importance_type}')
    
    def plot_shap_summary(self, X):
        """Plot SHAP summary"""
        features = self.model.extract_features(X)
        explainer = shap.TreeExplainer(self.model.xgb_model)
        shap_values = explainer.shap_values(features)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, features, feature_names=self.feature_names, show=False)
        self.save_plot(plt, 'shap_summary')
        
        # Additional SHAP plots
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, features, feature_names=self.feature_names, 
                         plot_type='bar', show=False)
        self.save_plot(plt, 'shap_importance_bar')
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix with percentages"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm / np.sum(cm),
            annot=True,
            fmt='.2%',
            cmap='Blues',
            square=True
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        self.save_plot(plt, 'confusion_matrix')
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        """Plot ROC curve with AUC"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        self.save_plot(plt, 'roc_curve')
    
    def plot_feature_interactions(self, X, top_n=3):
        """Plot top feature interactions using SHAP"""
        features = self.model.extract_features(X)
        explainer = shap.TreeExplainer(self.model.xgb_model)
        shap_values = explainer.shap_values(features)
        
        # Get top features by mean absolute SHAP value
        mean_abs_shap = np.abs(shap_values).mean(0)
        top_features = np.argsort(mean_abs_shap)[-top_n:]
        
        # Plot interactions for top features
        for idx in top_features:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(idx, shap_values, features, 
                               feature_names=self.feature_names,
                               show=False)
            self.save_plot(plt, f'interaction_{self.feature_names[idx]}')
    
    def save_performance_metrics(self, y_true, y_pred, y_pred_proba):
        """Save performance metrics to a text file"""

        metrics_path = os.path.join(self.save_dir, 'performance_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("Performance Metrics Summary\n")
            f.write("==========================\n\n")
            
            f.write("Classification Report:\n")
            f.write(classification_report(y_true, y_pred))
            
            f.write(f"\nAccuracy Score: {accuracy_score(y_true, y_pred):.4f}\n")
            f.write(f"F1 Score: {f1_score(y_true, y_pred):.4f}\n")
            
            # Add ROC AUC
            roc_auc = auc(*roc_curve(y_true, y_pred_proba)[:2])
            f.write(f"ROC AUC Score: {roc_auc:.4f}\n")

def train_hybrid_model(X_train, y_train, X_test, y_test, batch_size=64):
    """Main training function"""
    # Create save directory
    save_dir = './xgboost_conv_plots'
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data loaders
    train_size = int(0.8 * len(X_train))
    train_dataset = TensorDataset(X_train[:train_size], y_train[:train_size])
    val_dataset = TensorDataset(X_train[train_size:], y_train[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize and train hybrid model
    model = HybridModel()
    
    # Train CNN feature extractor with supervised learning
    print("Training CNN Feature Extractor...")
    model.train_cnn(train_loader, val_loader)
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    model.train_xgboost(X_train[:train_size], y_train[:train_size],
                       X_train[train_size:], y_train[train_size:])
    
    # Create analyzer and generate plots
    analyzer = ModelAnalyzer(model, model.feature_names, save_dir=save_dir)
    
    # Make predictions
    print("\nGenerating predictions...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Generate and save all analyses
    print("\nGenerating analysis plots...")
    analyzer.plot_feature_importance('weight')
    analyzer.plot_feature_importance('gain')
    analyzer.plot_feature_importance('cover')
    analyzer.plot_shap_summary(X_test)
    analyzer.plot_confusion_matrix(y_test, y_pred)
    analyzer.plot_roc_curve(y_test, y_pred_proba)
    analyzer.plot_feature_interactions(X_test)
    analyzer.save_performance_metrics(y_test, y_pred, y_pred_proba)
    
    return model, analyzer


def main():
    device = get_device()
    print(f"Using device: {device}")
    
    # Load and prepare data
    train_data = pd.read_csv('./training_data.csv')
    test_data = pd.read_csv('./test_data.csv')

    # Load and prepare data
    X_train = torch.FloatTensor(train_data.drop('label', axis=1).values)
    y_train = torch.FloatTensor(train_data['label'].values).reshape(-1, 1)
    X_test = torch.FloatTensor(test_data.drop('label', axis=1).values)
    y_test = torch.FloatTensor(test_data['label'].values).reshape(-1, 1)

    # Train the hybrid model
    model, analyzer = train_hybrid_model(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()