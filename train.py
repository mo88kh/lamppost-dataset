import torch, torchvision
from torchvision import transforms
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path
import time, json, numpy as np
from collections import Counter

# ---------------------- Parameters ----------------------
DATA_DIR   = Path("data/exports")  # Adjust path as needed
BS         = 64
LR         = 1e-3
EPOCHS     = 20
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------- Transforms ----------------------
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.25),
    transforms.ToTensor()
])
val_tf = transforms.ToTensor()

# ---------------------- Datasets ------------------------
train_ds = torchvision.datasets.ImageFolder(DATA_DIR/"train", transform=train_tf)
val_ds   = torchvision.datasets.ImageFolder(DATA_DIR/"val",   transform=val_tf)
test_ds  = torchvision.datasets.ImageFolder(DATA_DIR/"test",  transform=val_tf)

train_ld = torch.utils.data.DataLoader(train_ds, BS, shuffle=True,  num_workers=2)
val_ld   = torch.utils.data.DataLoader(val_ds,   BS, shuffle=False, num_workers=2)
test_ld  = torch.utils.data.DataLoader(test_ds,  BS, shuffle=False, num_workers=2)

print(f"train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")

# ---------------------- Model Setup ----------------------
model = torchvision.models.efficientnet_b0(weights=None, num_classes=2)
model.to(DEVICE)

# Weighted loss to address imbalance
counts = Counter([lbl for _, lbl in train_ds.samples])
total  = counts[0] + counts[1]
weights = torch.tensor([
    total / (2 * counts[0]),
    total / (2 * counts[1])
], dtype=torch.float32).to(DEVICE)

loss_fn = nn.CrossEntropyLoss(weight=weights)
optim   = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)

# ---------------------- Training Loop ----------------------
def run_epoch(loader, train=True):
    model.train(train)
    loss_sum, n = 0, 0
    metric = MulticlassAccuracy(num_classes=2).to(DEVICE)
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if train:
            optim.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        if train:
            loss.backward()
            optim.step()
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
        metric.update(logits, y)
    return loss_sum / n, metric.compute().item()

if __name__ == "__main__":
    # -------------- Setup --------------
    print(f"train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")

    best_val = 0
    best_state = None
    patience = 3
    wait = 0

    # -------------- Train loop --------------
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_ld, train=True)
        val_loss, val_acc = run_epoch(val_ld, train=False)
        scheduler.step()

        print(f"{epoch:02d} {tr_loss:.3f} {tr_acc:.3f} | "
              f"{val_loss:.3f} {val_acc:.3f}  ({time.time()-t0:.1f}s)")

        if val_acc > best_val:
            best_val = val_acc
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early-stop at epoch", epoch)
                break

    # -------------- Evaluate best model --------------
    model.load_state_dict(best_state)

    metric_acc = MulticlassAccuracy(num_classes=2).to(DEVICE)
    metric_cm  = MulticlassConfusionMatrix(num_classes=2).to(DEVICE)

    model.eval()
    with torch.no_grad():
        for x, y in test_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            metric_acc.update(logits, y)
            metric_cm.update(logits, y)

    print("Test accuracy:", metric_acc.compute().item())

    cm = metric_cm.compute().cpu().numpy()
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0,1], ["neg","pos"]); plt.yticks([0,1], ["neg","pos"])
    plt.colorbar(); plt.title("Confusion matrix"); plt.show()

    # -------------- Save ------------------
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(best_state, "models/efficientnet_best.pt")
    with open("models/final_metrics.json", "w") as f:
        json.dump({
            "accuracy": metric_acc.compute().item(),
            "class_counts": dict(counts),
            "best_val": best_val,
            "epochs_ran": epoch,
            "stopped_early": wait >= patience
        }, f, indent=2)
