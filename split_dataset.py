import os
import shutil
import random
from typing import List, Dict, Tuple

# ===================== CONFIGURAÇÕES =====================
# Caminhos (ajuste conforme solicitado)
SRC_DIR = r"C:\Users\Bezada\Desktop\TI - 6\PreprocessingResults"          # contém as pastas de classe
DST_DIR = r"C:\Users\Bezada\Desktop\TI - 6\DataSetProcessedSplited"      # será criado train/val/test

# Proporções do split (soma deve ser 1.0)
SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train, val, test

# Reprodutibilidade
RANDOM_SEED = 42

# Ação sobre arquivos: copy2 (preserva metadata) ou move
OPERATION = "copy"  # "copy" ou "move"

# Classes reconhecidas (com mapeamento de aliases para tolerar variações)
STANDARD_CLASSES = ["Benign cases", "Malignant cases", "Normal cases"]
CLASS_ALIASES = {
    "benign cases": "Benign cases",
    "bengin cases": "Benign cases",
    "benign": "Benign cases",
    "malignant cases": "Malignant cases",
    "malignant": "Malignant cases",
    "normal cases": "Normal cases",
    "normal": "Normal cases",
}
# Extensões aceitas
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
# ========================================================


def normalize_class_name(name: str) -> str:
    key = name.strip().lower()
    return CLASS_ALIASES.get(key, name)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]


def split_counts(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Converte proporções em contagens inteiras que somam n."""
    tr = int(round(n * ratios[0]))
    va = int(round(n * ratios[1]))
    te = n - tr - va
    # Ajuste final se arredondamento causar discrepância
    if te < 0:
        # corrige reduzindo do maior entre tr/va
        if tr >= va and tr > 0: 
            tr += te  # te é negativo
        elif va > 0:
            va += te
        te = 0
    return tr, va, te


def choose_op():
    if OPERATION.lower() == "move":
        return shutil.move
    return shutil.copy2


def main():
    random.seed(RANDOM_SEED)

    if not os.path.isdir(SRC_DIR):
        raise FileNotFoundError(f"Diretório de origem não existe: {SRC_DIR}")

    # 1) Descobrir pastas de classe na origem
    class_src_paths: Dict[str, str] = {}
    for entry in os.listdir(SRC_DIR):
        full = os.path.join(SRC_DIR, entry)
        if os.path.isdir(full):
            norm = normalize_class_name(entry)
            if norm in STANDARD_CLASSES:
                class_src_paths[norm] = full

    if not class_src_paths:
        raise RuntimeError(
            "Não encontrei pastas de classe válidas em SRC_DIR. "
            "Esperado algo como 'Benign cases', 'Malignant cases', 'Normal cases'."
        )

    print("Classes detectadas e normalizadas:")
    for cname, p in class_src_paths.items():
        print(f"  - {cname}: {p}")

    # 2) Criar estrutura de destino
    for split in ("train", "val", "test"):
        for cname in STANDARD_CLASSES:
            ensure_dir(os.path.join(DST_DIR, split, cname))

    # 3) Para cada classe, listar imagens, embaralhar e dividir
    summary = { "train": {}, "val": {}, "test": {} }
    do_file = choose_op()

    for cname in STANDARD_CLASSES:
        src_c = class_src_paths.get(cname)
        if not src_c:
            print(f"[AVISO] Classe não encontrada na origem, pulando: {cname}")
            continue

        files = list_images(src_c)
        if not files:
            print(f"[AVISO] Sem imagens na classe: {cname}")
            continue

        random.shuffle(files)

        n = len(files)
        n_train, n_val, n_test = split_counts(n, SPLIT_RATIOS)

        splits = {
            "train": files[:n_train],
            "val":   files[n_train:n_train+n_val],
            "test":  files[n_train+n_val:]
        }

        # 4) Copiar/mover
        for split_name, fnames in splits.items():
            dst_c = os.path.join(DST_DIR, split_name, cname)
            ensure_dir(dst_c)
            for fname in fnames:
                src_path = os.path.join(src_c, fname)
                dst_path = os.path.join(dst_c, fname)
                do_file(src_path, dst_path)

            summary[split_name][cname] = len(fnames)

        # Log por classe
        print(f"\nClasse: {cname}")
        print(f"  Total: {n}")
        print(f"    → train: {n_train}")
        print(f"    → val:   {n_val}")
        print(f"    → test:  {n_test}")

    # 5) Resumo final
    print("\n===== RESUMO FINAL =====")
    for split in ("train", "val", "test"):
        total_split = sum(summary[split].get(c, 0) for c in STANDARD_CLASSES)
        print(f"{split.upper()} (total {total_split}):")
        for cname in STANDARD_CLASSES:
            print(f"  {cname}: {summary[split].get(cname, 0)}")

    print(f"\nSplit concluído em: {DST_DIR}")
    print(f"Ação realizada: {OPERATION.upper()} (originais preservados se COPY)")
    print(f"Seed: {RANDOM_SEED}, Proporções: train/val/test = {SPLIT_RATIOS}")


if __name__ == "__main__":
    main()
