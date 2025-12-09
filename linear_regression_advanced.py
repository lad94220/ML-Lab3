import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.model_selection import KFold

from loss import GAR
from utils import bike_sharing, pair_dataset, protein_data, set_all_seeds


class LinearRegression(nn.Module):
    """A simple Linear Regression model."""

    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # The MLP model in models.py returns (output, features)
        # We return the output twice to maintain compatibility with the training loop structure.
        out = self.linear(x)
        return out, out


def evaluate(model, dataloader, device, num_targets):
    """Evaluates the model on the given dataloader and returns metrics."""
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for data in dataloader:
            X, Y = data[0].to(device), data[1].to(device)
            pred_Y, _ = model(X)
            preds.append(pred_Y.cpu().numpy())
            truths.append(Y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    truths = np.concatenate(truths, axis=0)

    maes, rmses, pearsons, spearmans = [], [], [], []
    for i in range(num_targets):
        pred, truth = preds[:, i], truths[:, i]
        maes.append(np.abs(pred - truth).mean())
        rmses.append(((pred - truth) ** 2).mean() ** 0.5)
        # Handle cases where variance is zero, which would make corrcoef return nan
        if np.std(truth) > 0 and np.std(pred) > 0:
            pearsons.append(np.corrcoef(truth, pred, rowvar=False)[0, 1])
            spearmans.append(stats.spearmanr(truth, pred).statistic)
        else:
            pearsons.append(0.0)
            spearmans.append(0.0)

    return np.mean(maes), np.mean(rmses), np.mean(pearsons), np.mean(spearmans)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Regression on noisy datasets")
    parser.add_argument(
        "--dataset",
        default="bike_sharing",
        type=str,
        choices=["bike_sharing", "protein"],
        help="The name for the dataset to use from 'advanced-data' folder.",
    )
    parser.add_argument("--lr", default=0.01, type=float, help="Initial learning rate")
    parser.add_argument(
        "--decay", default=1e-4, type=float, help="Weight decay for the optimizer"
    )
    parser.add_argument("--batch_size", default=256, type=int, help="Training batch size")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train")
    args = parser.parse_args()

    # --- Configuration ---
    SEED = 123
    set_all_seeds(SEED)
    
    # Set device
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        print(f'Using Intel XE Graphics (XPU): {torch.xpu.get_device_name(0)}')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using CUDA GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    # --- Data Loading ---
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "bike_sharing":
        # Assuming the noisy data was created by advanced-data.py
        path = os.path.join("advanced-data", "bike_sharing_noisy.npz")
        trX, trY, teX, teY = bike_sharing(path=path)
    elif args.dataset == "protein":
        path = os.path.join("advanced-data", "protein_noisy.npz")
        trX, trY, teX, teY = protein_data(path=path)

    num_targets = trY.shape[1]
    print(f"Train shapes: X-{trX.shape}, Y-{trY.shape}")
    print(f"Test shapes:  X-{teX.shape}, Y-{teY.shape}")

    tr_pair_data = pair_dataset(trX, trY)
    te_pair_data = pair_dataset(teX, teY)
    testloader = torch.utils.data.DataLoader(
        dataset=te_pair_data, batch_size=args.batch_size, shuffle=False
    )

    # --- Training & Evaluation ---
    print("\n" + "-" * 30)
    print("Start Training with Linear Regression")
    print("-" * 30)

    model = LinearRegression(input_dim=trX.shape[-1], output_dim=num_targets).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay
    )
    milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )
    loss_fn = nn.MSELoss()  # Using standard Mean Squared Error for regression

    trainloader = torch.utils.data.DataLoader(
        dataset=tr_pair_data, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            tr_X, tr_Y = data[0].to(device), data[1].to(device)
            pred_Y, _ = model(tr_X)

            loss = loss_fn(pred_Y, tr_Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_epoch_loss = epoch_loss / (i + 1)
        epoch_time = time.time() - start_time

        # --- Evaluation at end of epoch ---
        train_mae, train_rmse, train_pearson, train_spearman = evaluate(
            model, trainloader, device, num_targets
        )
        test_mae, test_rmse, test_pearson, test_spearman = evaluate(
            model, testloader, device, num_targets
        )

        print(f"Epoch: {epoch+1:03}/{args.epochs} | Time: {epoch_time:.2f}s | LR: {scheduler.get_last_lr()[0]:.4f}")
        print(f"\tTrain Loss: {avg_epoch_loss:.4f} | MAE: {train_mae:.4f} | RMSE: {train_rmse:.4f} | Pearson: {train_pearson:.4f}")
        print(f"\tTest  MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | Pearson: {test_pearson:.4f} | Spearman: {test_spearman:.4f}")

    print("\n" + "=" * 50)
    print("Final Test Metrics")
    print("=" * 50)
    final_mae, final_rmse, final_pearson, final_spearman = evaluate(
        model, testloader, device, num_targets
    )
    print(f"MAE: {final_mae:.4f}")
    print(f"RMSE: {final_rmse:.4f}")
    print(f"Pearson Correlation: {final_pearson:.4f}")
    print(f"Spearman Correlation: {final_spearman:.4f}")
    print("=" * 50)