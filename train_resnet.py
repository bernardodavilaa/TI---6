# train_resnet.py
import os
import json
import time
import random
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# ======================= CONFIG Padrão =======================
DATA_ROOT = r"C:\Users\Bezada\Desktop\TI - 6\DataSetProcessedSplited"  # contém train/, val/, test/
OUTPUT_DIR = "./outputs_resnet"                                        # onde salvar modelos e métricas
#como rodar: python train_resnet.py --data_root "C:\Users\Bezada\Desktop\TI - 6\DataSetProcessedSplited" --out ".\outputs_resnet"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 7                 # early stopping (épocas sem melhora em val_loss)
FINE_TUNE_LAST_BLOCK = True  # True: descongela only layer4 + fc; False: feature extractor (congela tudo)
SEED = 42
# ============================================================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # (opcional) determinismo:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(imagefolder_dataset: datasets.ImageFolder) -> torch.Tensor:
    """Calcula pesos de classe inversamente proporcionais à frequência (para CrossEntropyLoss)."""
    targets = [label for _, label in imagefolder_dataset.samples]
    counts = np.bincount(targets)
    counts = counts.astype(np.float32)
    weights = 1.0 / np.clip(counts, a_min=1.0, a_max=None)
    weights = weights * (len(counts) / weights.sum())  # normaliza leve
    return torch.tensor(weights, dtype=torch.float32)


def get_transforms(img_size=224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, eval_tf


def build_dataloaders(data_root, batch_size, num_workers):
    train_tf, eval_tf = get_transforms(IMG_SIZE)

    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_root, "val"),   transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(data_root, "test"),  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def build_model(num_classes=3, fine_tune_last_block=True, device="cuda"):
    # ResNet18 com pesos ImageNet (API nova do torchvision)
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if fine_tune_last_block:
        # Congela tudo
        for p in model.parameters():
            p.requires_grad = False
        # Descongela somente layer4 e fc
        for p in model.layer4.parameters():
            p.requires_grad = True
        # Troca a última camada (fc)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        # fc por padrão requires_grad=True
    else:
        # Feature extractor puro: congela tudo e só treina a fc
        for p in model.parameters():
            p.requires_grad = False
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)

    model = model.to(device)
    return model


@dataclass
class TrainState:
    best_val_loss: float = float("inf")
    best_epoch: int = -1


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    acc = accuracy_score(y_true, y_pred)
    return epoch_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return epoch_loss, acc, prec, rec, f1, y_true, y_pred


def plot_confusion_matrix(cm, classes, out_path):
    fig = plt.figure(figsize=(6, 5), dpi=140)
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de Confusão")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("Verdadeiro")
    plt.xlabel("Predito")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(
    data_root=DATA_ROOT, output_dir=OUTPUT_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
    patience=PATIENCE, fine_tune_last_block=FINE_TUNE_LAST_BLOCK, seed=SEED
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")  

    # Loaders
    global IMG_SIZE
    IMG_SIZE = img_size  # garantir que get_transforms use o valor passado
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(
        data_root, batch_size, num_workers
    )

    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print("Classes:", class_to_idx)

    # Pesos por classe
    class_weights = compute_class_weights(train_ds).to(device)
    print("Class weights:", class_weights.detach().cpu().numpy().round(3))

    # Modelo
    model = build_model(num_classes=len(class_to_idx), fine_tune_last_block=fine_tune_last_block, device=device)

    # Loss, otimizador, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Otimiza somente parâmetros com grad
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_update, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # Treino + Early stopping
    state = TrainState()
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        dt = time.time() - t0
        print(f"[Epoch {epoch:02d}/{epochs}] "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | {dt:.1f}s")

        # Checkpoint do melhor val_loss
        if val_loss < state.best_val_loss - 1e-6:
            state.best_val_loss = val_loss
            state.best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            with open(os.path.join(output_dir, "class_to_idx.json"), "w", encoding="utf-8") as f:
                json.dump(class_to_idx, f, indent=2, ensure_ascii=False)
            print(f"  ↳ Novo melhor modelo salvo (época {epoch})")

        # Early stopping
        if epoch - state.best_epoch >= patience:
            print(f"Early stopping (sem melhora em {patience} épocas). Melhor época: {state.best_epoch}")
            break

    # Carrega melhor modelo
    best_model = build_model(num_classes=len(class_to_idx), fine_tune_last_block=fine_tune_last_block, device=device)
    best_model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth"), map_location=device))

    # Avaliação final em VAL (com melhor modelo, para registro)
    val_loss, val_acc, val_prec, val_rec, val_f1, y_true_val, y_pred_val = evaluate(best_model, val_loader, criterion, device)

    # Avaliação em TEST (final)
    test_loss, test_acc, test_prec, test_rec, test_f1, y_true, y_pred = evaluate(best_model, test_loader, criterion, device)
    report = classification_report(y_true, y_pred, target_names=[idx_to_class[i] for i in range(len(idx_to_class))], zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Salva métricas e artefatos
    metrics = {
        "val": {
            "loss": float(val_loss), "acc": float(val_acc),
            "precision_macro": float(val_prec), "recall_macro": float(val_rec), "f1_macro": float(val_f1),
        },
        "test": {
            "loss": float(test_loss), "acc": float(test_acc),
            "precision_macro": float(test_prec), "recall_macro": float(test_rec), "f1_macro": float(test_f1),
        },
        "best_epoch": state.best_epoch,
        "history": history,
        "classes": class_to_idx
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    plot_confusion_matrix(cm, [idx_to_class[i] for i in range(len(idx_to_class))],
                          os.path.join(output_dir, "confusion_matrix.png"))

    print("\n==== RESULTADOS ====")
    print(f"Melhor época: {state.best_epoch}")
    print(f"VAL  -> loss: {val_loss:.4f} | acc: {val_acc:.4f} | F1(macro): {val_f1:.4f}")
    print(f"TEST -> loss: {test_loss:.4f} | acc: {test_acc:.4f} | F1(macro): {test_f1:.4f}")
    print("\nClassification Report (TEST):\n", report)
    print(f"\nArtefatos salvos em: {os.path.abspath(output_dir)}")
    print("Arquivos: best_model.pth, class_to_idx.json, metrics.json, classification_report.txt, confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--out", type=str, default=OUTPUT_DIR)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--finetune_last_block", action="store_true", default=FINE_TUNE_LAST_BLOCK)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    main(
        data_root=args.data_root, output_dir=args.out, img_size=args.img_size,
        batch_size=args.batch_size, num_workers=args.num_workers, epochs=args.epochs,
        lr=args.lr, weight_decay=args.weight_decay, patience=args.patience,
        fine_tune_last_block=args.finetune_last_block, seed=args.seed
    )
