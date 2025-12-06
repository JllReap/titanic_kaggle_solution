import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, accuracy_score
from torch.nn.functional import cross_entropy

from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class DNN(nn.Module):

    def __init__(self, output_size=1, epochs=100, batch_size=32, learning_rate=0.001,
                 embedding_size=128, device='cpu', kd_alpha=0, early_stopping_patience=10):
        super(DNN, self).__init__()

        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.kd_alpha = kd_alpha
        self.early_stopping_patience = early_stopping_patience

        self.output_size = output_size
        self.embeddings = None
        self.scaler = None
        self.input_size = 0
        self.cat_column_names = []

    def _build_model(self):
        """Build model for the custom input size."""
        self.embedding_mlp = nn.Embedding(sum(self.field_dims), self.embedding_size)

        num_numerical_features = len(self.num_columns)
        input_dim = (len(self.field_dims) * self.embedding_size) + num_numerical_features

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, self.output_size),
        )

    def forward(self, x_num, x_cat, return_embeddings=False):
        """Forward function."""

        mlp_embed_x = self.embedding_mlp(x_cat)
        mlp_embed_x = mlp_embed_x.view(mlp_embed_x.size(0), -1)
        mlp_x = torch.cat([mlp_embed_x, x_num], dim=1)
        mlp_out = self.mlp(mlp_x)

        if return_embeddings:
            embeddings = self.mlp[:-1](mlp_x)
            return mlp_out, embeddings
        else:
            return mlp_out

    def fit(self, X_train: pd.DataFrame, Y_train: pd.DataFrame, X_val: pd.DataFrame, Y_val: pd.DataFrame):
        """Training."""

        # Distinct categorical and numerical columns.
        self.cat_column_names = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_columns = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
        self.field_dims = [X_train[col].nunique() for col in self.cat_column_names]

        X_train_num = X_train[self.num_columns]
        X_val_num = X_val[self.num_columns]

        # Build the model.

        self._build_model()
        self.to(self.device)

        # Process numerical columns using quantile transformer for all features except the OOF one.

        self.scaler = StandardScaler()
        self.scaler.fit(X_train_num.values)

        X_train_num_scaled = torch.tensor(self.scaler.transform(X_train_num.values)).float().to(self.device)
        X_val_num_scaled = torch.tensor(self.scaler.transform(X_val_num)).float().to(self.device)

        # Process categorical columns.

        X_train_cat = X_train[self.cat_column_names].astype('category').apply(lambda x: x.cat.codes)
        X_val_cat = X_val[self.cat_column_names].astype('category').apply(lambda x: x.cat.codes)

        X_train_cat_tensor = torch.tensor(X_train_cat.values).long().to(self.device)
        X_val_cat_tensor = torch.tensor(X_val_cat.values).long().to(self.device)

        # Create object for training.

        Y_train_tensor = torch.tensor(Y_train.values).float().to(self.device).unsqueeze(-1)
        Y_val_tensor = torch.tensor(Y_val.values).float().to(self.device).unsqueeze(-1)

        train_dataset = TensorDataset(X_train_num_scaled, X_train_cat_tensor, Y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val_num_scaled, X_val_cat_tensor, Y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-8)

        scaler = GradScaler()

        best_val_loss = np.inf
        patience_counter = 0

        # Train loop.

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            self.train()
            train_loss = 0.0
            train_batches = tqdm(train_loader, desc="Train Batches", leave=False)

            for batch_idx, (batch_X_num, batch_X_cat, batch_Y) in enumerate(train_batches):
                batch_X_num, batch_X_cat, batch_Y = batch_X_num.to(self.device), batch_X_cat.to(
                    self.device), batch_Y.to(self.device)

                optimizer.zero_grad()

                with autocast(): # mixed precision
                    outputs = self.forward(batch_X_num, batch_X_cat)
                    loss = criterion(outputs, batch_Y)

                # optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                train_batches.set_postfix({'Current Train Loss': loss.item()})

            train_loss /= len(train_loader)
            print(f"Training Loss: {train_loss:.4f}", end=' ')

            self.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            val_batches = tqdm(val_loader, desc="Val Batches", leave=False)

            with torch.no_grad():
                for batch_idx, (batch_X_num, batch_X_cat, batch_Y) in enumerate(val_batches):
                    batch_X_num, batch_X_cat, batch_Y = batch_X_num.to(self.device), batch_X_cat.to(
                        self.device), batch_Y.to(self.device)

                    with autocast():
                        outputs = self.forward(batch_X_num, batch_X_cat)
                        loss = criterion(outputs, batch_Y)

                    val_loss += loss.item()
                    # outputs = torch.sigmoid(outputs)
                    # outputs = (outputs > 0.5).long().cpu()
                    all_preds.append(outputs.cpu())
                    all_labels.append(batch_Y.cpu())
                    val_batches.set_postfix({'Current Val Loss': loss.item()})

            val_loss /= len(val_loader)
            all_preds = torch.cat(all_preds).squeeze()
            all_labels = torch.cat(all_labels).squeeze()

            probs = torch.sigmoid(all_preds)
            preds = (probs > 0.5).long()
            val_acc = (preds == all_labels.long()).float().mean().item()
            print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")

            if val_acc < best_val_loss:
                best_val_loss = val_acc
                patience_counter = 0
                self.save('checkpoints/tmp.pt')
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                break

            scheduler.step()

        self.load('checkpoints/tmp.pt')

    def predict(self, X: pd.DataFrame, return_embeddings=False):
        """Inference."""

        self.eval()
        with torch.no_grad():
            X_num = X[self.num_columns]
            X_num_scaled = torch.tensor(self.scaler.transform(X_num)).float().to(self.device)

            X_cat = X[self.cat_column_names].astype('category').apply(lambda x: x.cat.codes).values
            X_cat_tensor = torch.tensor(X_cat).long().to(self.device)

            dataset = TensorDataset(X_num_scaled, X_cat_tensor)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            predictions, total_embeddings = [], []
            print('Prediction')
            for X_num_batch, X_cat_batch in tqdm(data_loader):
                if return_embeddings:
                    preds, embeddings = self.forward(X_num_batch, X_cat_batch, return_embeddings)
                    predictions.append(preds.cpu().numpy())
                    total_embeddings.append(embeddings.cpu().numpy())
                else:
                    preds = self.forward(X_num_batch, X_cat_batch, return_embeddings)
                    predictions.append(preds.cpu().numpy())

            if return_embeddings:
                return np.squeeze(np.vstack(predictions)), np.squeeze(np.vstack(total_embeddings))
            else:
                return np.squeeze(np.vstack(predictions))

    def save(self, file_path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'scaler': self.scaler,
        }
        joblib.dump(checkpoint, file_path)

    def load(self, file_path: str):
        """Load model from checkpoint."""
        checkpoint = joblib.load(file_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
