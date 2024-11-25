import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn

def prepare_data(df1, df2):
    # Combine datasets and add a label for ventral/dorsal
    df1['region'] = 'ventral'
    df2['region'] = 'dorsal'
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    all_features = []
    all_labels = []
    
    for idx, row in combined_df.iterrows():
        # Calculate log10(FR) and log10(STTC) for each neuron
        if 'firing_rates' in row and 'sttc_values' in row:
            log10_FRs = np.log10(np.array(row['firing_rates']) + 1e-10)
            log10_STTCs = np.log10(np.array(row['sttc_values']) + 1e-10)
        else:
            raise ValueError(f"'firing_rates' or 'sttc_values' not found in row {idx}")
        
        # Extract waveform features for each neuron
        if 'neuron_data' in row:
            for neuron_id, neuron_info in row['neuron_data'].items():
                features = waveform_feature(neuron_info['template'])
                neuron_features = [
                    log10_FRs[int(neuron_id)],
                    log10_STTCs[int(neuron_id)],
                    features['trough_to_peak'],
                    features['fwhm'],
                    features['amplitude']
                ] + list(features['waveform_normalized'])
                
                # Check for NaN values
                if not np.any(np.isnan(neuron_features)):
                    all_features.append(neuron_features)
                    all_labels.append(1 if row['region'] == 'dorsal' else 0)
                else:
                    print(f"Skipping neuron {neuron_id} in row {idx} due to NaN values")
        else:
            raise ValueError(f"'neuron_data' not found in row {idx}")
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    # Additional check for NaN values
    valid_indices = ~np.isnan(X).any(axis=1)
    X = X[valid_indices]
    y = y[valid_indices]
    
    print(f"Shape of X after NaN removal: {X.shape}")
    print(f"Shape of y after NaN removal: {y.shape}")
    
    # Perform stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

class SNNComponent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, beta=0.95):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x, num_steps=100):
        spk_rec = []
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)

        return torch.stack(spk_rec, dim=0)

class TraditionalNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class WaveformCNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 50, output_size)  # Assuming waveform length of 100

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class HybridModel(nn.Module):
    def __init__(self, input_size, snn_hidden_size, nn_hidden_size, cnn_output_size, output_size):
        super().__init__()
        self.snn = SNNComponent(5, snn_hidden_size, output_size)  # For log10_FR, log10_STTC, and 3 waveform features
        self.nn = TraditionalNN(5, nn_hidden_size, output_size)   # For log10_FR, log10_STTC, and 3 waveform features
        self.cnn = WaveformCNN(1, cnn_output_size)  # For normalized waveform data
        self.fc_combine = nn.Linear(output_size * 2 + cnn_output_size, output_size)

    def forward(self, x):
        # Split input into stats, waveform features, and normalized waveform
        stats_and_features = x[:, :5]  # First 5 columns are log10_FR, log10_STTC, and 3 waveform features
        waveform = x[:, 5:].unsqueeze(1)  # Rest is normalized waveform data, add channel dimension

        snn_out = self.snn(stats_and_features).mean(dim=0)  # Average over time steps
        nn_out = self.nn(stats_and_features)
        cnn_out = self.cnn(waveform)
        combined = torch.cat((snn_out, nn_out, cnn_out), dim=1)
        return self.fc_combine(combined)

def train_hybrid_model(X_train, y_train, X_test, y_test, input_size, epochs=100, batch_size=32):
    model = HybridModel(input_size=input_size, snn_hidden_size=64, nn_hidden_size=32, cnn_output_size=16, output_size=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_loss = criterion(test_outputs, y_test_tensor)
                preds = torch.sigmoid(test_outputs) > 0.5
                accuracy = (preds.float() == y_test_tensor).float().mean()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Accuracy: {accuracy.item():.4f}')

    return model

def evaluate_model(model, X_test, y_test):
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        preds = torch.sigmoid(test_outputs) > 0.5
        accuracy = (preds.float() == y_test_tensor).float().mean()
        
    print(f'Test Accuracy: {accuracy.item():.4f}')
    return accuracy.item()

# Usage
try:
    X_train, X_test, y_train, y_test = prepare_data(df1, df2)
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_test: {y_test.shape}")
    
    # Check for any remaining NaN values
    print(f"NaN values in X_train: {np.isnan(X_train).any()}")
    print(f"NaN values in X_test: {np.isnan(X_test).any()}")
    
    # Print some statistics about the data
    print(f"X_train mean: {np.mean(X_train)}, std: {np.std(X_train)}")
    print(f"y_train mean: {np.mean(y_train)}, std: {np.std(y_train)}")
    
    # Check class balance
    print(f"Class balance in y_train: {np.bincount(y_train.astype(int))}")
    print(f"Class balance in y_test: {np.bincount(y_test.astype(int))}")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check your input data and ensure it contains the necessary information.")


def evaluate_model_comprehensive(model, X, y, device='cpu'):
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
        predictions = (probabilities > 0.5).astype(int)
        
    # 1. Accuracy
    accuracy = (predictions == y).mean()
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y, predictions)
    
    # 3. Precision, Recall, F1-score
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # 4. ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y, probabilities)
    roc_auc = auc(fpr, tpr)
    
    # 5. Precision-Recall Curve and Average Precision
    precision_curve, recall_curve, _ = precision_recall_curve(y, probabilities)
    average_precision = average_precision_score(y, probabilities)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'average_precision': average_precision,
        'probabilities': probabilities
    }

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall_curve(recall, precision, average_precision):
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
    plt.show()

def cross_validate(model_class, X, y, n_splits=5, epochs=100, batch_size=32, device='cpu'):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{n_splits}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        input_size = X_train.shape[1]
        snn_hidden_size = 64
        nn_hidden_size = 32
        cnn_output_size = 16
        output_size = 1
        
        model = model_class(
            input_size=input_size,
            snn_hidden_size=snn_hidden_size,
            nn_hidden_size=nn_hidden_size,
            cnn_output_size=cnn_output_size,
            output_size=output_size
        ).to(device)
        
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            model.train()
            for i in range(0, len(X_train), batch_size):
                batch_X = torch.FloatTensor(X_train[i:i+batch_size]).to(device)
                batch_y = torch.FloatTensor(y_train[i:i+batch_size]).unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                val_results = evaluate_model_comprehensive(model, X_val, y_val, device)
                print(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {val_results['accuracy']:.4f}")
        
        final_results = evaluate_model_comprehensive(model, X_val, y_val, device)
        cv_scores.append(final_results['accuracy'])
        print(f"Fold {fold + 1} Accuracy: {final_results['accuracy']:.4f}")
    
    print(f"Cross-validation complete. Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Usage example
def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_class, epochs=100, batch_size=32, device='cpu'):
    input_size = X_train.shape[1]
    snn_hidden_size = 64  # You can adjust these values as needed
    nn_hidden_size = 32
    cnn_output_size = 16
    output_size = 1  # For binary classification

    model = model_class(
        input_size=input_size,
        snn_hidden_size=snn_hidden_size,
        nn_hidden_size=nn_hidden_size,
        cnn_output_size=cnn_output_size,
        output_size=output_size
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = torch.FloatTensor(X_train[i:i+batch_size]).to(device)
            batch_y = torch.FloatTensor(y_train[i:i+batch_size]).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / (len(X_train) // batch_size))
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_test).to(device))
            val_loss = criterion(val_outputs, torch.FloatTensor(y_test).unsqueeze(1).to(device))
            val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
    
    # Final evaluation
    test_results = evaluate_model_comprehensive(model, X_test, y_test, device)
    
    # Plotting
    plot_training_history(train_losses, val_losses)
    plot_roc_curve(test_results['fpr'], test_results['tpr'], test_results['roc_auc'])
    plot_precision_recall_curve(test_results['recall_curve'], test_results['precision_curve'], test_results['average_precision'])
    
    return model, test_results

# Usage
try:
    model, results = train_and_evaluate_model(X_train, y_train, X_test, y_test, HybridModel)
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-score: {results['f1_score']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")

    # Perform cross-validation
    cross_validate(HybridModel, np.vstack((X_train, X_test)), np.concatenate((y_train, y_test)))
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check your input data and ensure it contains the necessary information.")