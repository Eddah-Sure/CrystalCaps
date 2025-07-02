import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter, to_dense_batch
from sklearn.metrics import mean_absolute_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Dataloader import GraphDataset
from Model import CGNe3


def train_CGN( model, train_loader, optimizer, device, epochs, scheduler, early_stopping_patience, checkpoint_path, metrics_path):
    def run_caps_epoch(model, loader, optimizer=None):
        is_train = optimizer is not None
        model.train() if is_train else model.eval()
        total_loss, total_mae = 0.0, 0.0

        for batch in loader:
            batch = batch.to(device)
            if is_train:
                optimizer.zero_grad()
            output = model(batch)
            loss = F.mse_loss(output, batch.y)
            mae = F.l1_loss(output, batch.y)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            total_mae += mae.item() * batch.num_graphs

        size = len(loader.dataset)
        return total_loss / size, total_mae / size

    dataset = train_loader.dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_data, batch_size=train_loader.batch_size, shuffle=True, collate_fn=Batch.from_data_list)
    val_loader = DataLoader(val_data, batch_size=train_loader.batch_size, collate_fn=Batch.from_data_list)

    best_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    metrics = { 'epoch': [], 'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': [], 'learning_rate': []}
    pd.DataFrame(columns=metrics.keys()).to_csv(metrics_path, index=False)

    for epoch in range(epochs):
        train_loss, train_mae = run_caps_epoch(model, train_loader, optimizer)
        val_loss, val_mae = run_caps_epoch(model, val_loader)
        lr = optimizer.param_groups[0]['lr']
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['train_mae'].append(train_mae)
        metrics['val_loss'].append(val_loss)
        metrics['val_mae'].append(val_mae)
        metrics['learning_rate'].append(lr)

        pd.DataFrame({k: [v[-1]] for k, v in metrics.items()}).to_csv( metrics_path, mode='a', header=False, index=False)

        if scheduler:
            scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss, }, checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    return model, train_losses, val_losses, metrics

def evaluate_CGN(model, test_loader, device, results_path=None):
    model.eval()
    total_mse = 0
    total_mae = 0
    predictions, targets, material_ids = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            target = batch.y
            total_mse += F.mse_loss(output, target, reduction='sum').item()
            total_mae += F.l1_loss(output, target, reduction='sum').item()
            predictions.extend(output.cpu().numpy().flatten())
            targets.extend(target.cpu().numpy().flatten())

            if hasattr(batch, 'material_id'):
                ids = batch.material_id if isinstance(batch.material_id, list) else [batch.material_id[i] for i in range(batch.num_graphs)]
                material_ids.extend(ids)

    num_samples = len(test_loader.dataset)
    mse = total_mse / num_samples
    mae = total_mae / num_samples
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)

    results = { "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "predictions": predictions, "targets": targets, "material_ids": material_ids }
    if results_path:
        pd.DataFrame([{ "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}]).to_csv(results_path, index=False)
    return results

def run_CGN(dataset_path, target_name, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    target_name = str(target_name)
    os.makedirs(f"")
    metrics_path = os.path.join(f"")
    checkpoint_path = os.path.join(f"")
    test_metrics_path = os.path.join(f"")
    results_file = os.path.join(f"")

    dataset = GraphDataset(dataset_path, target_name=target_name)
  
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"Split: {train_size} train, {test_size} test")

    batch_size = batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=Batch.from_data_list)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=Batch.from_data_list)

    print("Initializing Spatial-Aware GNN + Capsule Network model")
    model = CGNe3(
        node_features=dataset[0].x.size(1),
        edge_features=dataset[0].edge_attr.size(1),
        hidden_channels=hidden_channels,
        num_conv_layers=num_conv_layers,
        primary_caps=primary_caps,
        primary_dim=primary_dim,
        secondary_caps=secondary_caps,
        secondary_dim=secondary_dim,
        dropout_rate=dropout_rate
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=True, min_lr=1e-6
    )

    model, train_losses, val_losses, _ = train_CGN(
        model, train_loader, optimizer, device,
        epochs=epochs,
        scheduler=scheduler,
        early_stopping_patience,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path
    )
    test_results = evaluate_CGN(model, test_loader, device, results_path=test_metrics_path)
    results_df = pd.DataFrame({
        'material_id': test_results['material_ids'],
        'Actual': test_results['targets'],
        'Predicted': test_results['predictions'],
        'Absolute Erroe': np.abs(np.array(test_results['predictions']) - np.array(test_results['targets']))
    })
    results_df.to_csv(results_file, index=False)
    print(f"Prediction results saved to {results_file}")
    return model, test_results

if __name__ == "__main__":
    dataset_path = ""
    model, results = run_CGN(
        dataset_path,
        target_name="",
        epochs=No of epochs
    )
