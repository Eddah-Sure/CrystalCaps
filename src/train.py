import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter, to_dense_batch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_error
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_spatial_gnn(
    model, train_loader, optimizer, device,
    epochs=100, scheduler=None, early_stopping_patience=30,
    checkpoint_path="best_model.pt", metrics_path="training_metrics.csv"
):
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

    print(f"CapsGNN: Training on {train_size}, validating on {val_size}")

    best_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    metrics = {
        'epoch': [], 'train_loss': [], 'train_mae': [],
        'val_loss': [], 'val_mae': [], 'learning_rate': []
    }
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

        pd.DataFrame({k: [v[-1]] for k, v in metrics.items()}).to_csv(
            metrics_path, mode='a', header=False, index=False
        )

        if scheduler:
            scheduler.step(val_loss)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f} | "
              f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, LR: {lr:.2e}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            print(f"Saved best model at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f" Loaded best model from epoch {checkpoint['epoch']+1} with loss {checkpoint['loss']:.6f}")

    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    return model, train_losses, val_losses, metrics


def evaluate_spatial_gnn(model, test_loader, device, results_path=None):
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

    results = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": predictions,
        "targets": targets,
        "material_ids": material_ids
    }

    if results_path:
        pd.DataFrame([{
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }]).to_csv(results_path, index=False)
      

    return results

def run_spatial_gnn_capsnet(dataset_path, target_name, epochs, pretrained_model_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_name = str(target_name)

    dataset = CartesianGraphDataset(dataset_path, target_name=target_name)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
 
    # Data loaders
    batch_size = batch_size`
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=Batch.from_data_list)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=Batch.from_data_list)

    print("Initializing Spatial-Aware GNN + Capsule Network model")
    model = CartesianAwareCrystalGNNCapsNet(
        node_features=dataset[0].x.size(1),
        edge_features=dataset[0].edge_attr.size(1),
        hidden_channels=hidden_channels,
        num_conv_layers=num_conv_layers,
        primary_caps=primary_caps,
        primary_dim=primary_dim,
        secondary_caps=secondary_caps,
        secondary_dim=secondary_dim,
        dropout_rate=dropout_rate #optional
    ).to(device)


    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}")
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Pretrained model loaded.")
    else:
        print("No pretrained model provided. Training from scratch.")


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
   

    # Weight initialization
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=True, min_lr=1e-6
    )

    # Train
    model, train_losses, val_losses, _ = train_spatial_gnn(
        model, train_loader, optimizer, device,
        epochs=epochs,
        scheduler=scheduler,
        early_stopping_patience=50,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path
    )

    # Evaluate
    test_results = evaluate_spatial_gnn(model, test_loader, device, results_path=test_metrics_path)

    return model, test_results
