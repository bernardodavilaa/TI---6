# preprocessing_gui.py
import os
import random
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageEnhance, ImageTk
import traceback

try:
    import numpy as np
except:
    np = None

# -------- Utilidades --------
VALID_CLASS_ALIASES = {
    # mapeia várias grafias comuns para nomes padronizados
    "benign cases": "Benign cases",
    "bengin cases": "Benign cases",
    "benign": "Benign cases",
    "malignant cases": "Malignant cases",
    "malignant": "Malignant cases",
    "normal cases": "Normal cases",
    "normal": "Normal cases",
}

STANDARD_CLASSES = ["Benign cases", "Malignant cases", "Normal cases"]

def normalize_class_name(name: str) -> str:
    key = name.strip().lower()
    return VALID_CLASS_ALIASES.get(key, name)

def find_class_folders(base_dir):
    """
    Varre subpastas do base_dir e tenta identificar classes padrão.
    Retorna dict {class_name: full_path}
    """
    out = {}
    if not base_dir or not os.path.isdir(base_dir):
        return out
    for entry in os.listdir(base_dir):
        full = os.path.join(base_dir, entry)
        if os.path.isdir(full):
            norm = normalize_class_name(entry)
            if norm in STANDARD_CLASSES:
                out[norm] = full
    return out

def list_images(folder):
    if not os.path.isdir(folder):
        return []
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            files.append(f)
    return files

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def safe_open_rgb(path, target_size=None, to_rgb=True):
    img = Image.open(path)
    if to_rgb:
        img = img.convert("RGB")
    if target_size:
        img = img.resize(target_size, Image.BILINEAR)
    return img

# -------- Augmentations --------
def augment_random(
    img: Image.Image,
    rot_deg=25,
    do_hflip=True,
    do_rot=True,
    do_contrast=True,
    do_brightness=True,
    do_sharpness=True,
    do_noise=True,
):
    ops = []

    if do_hflip:
        ops.append(lambda im: im.transpose(Image.FLIP_LEFT_RIGHT))
    if do_rot:
        ops.append(lambda im: im.rotate(random.uniform(-rot_deg, rot_deg)))
    if do_contrast:
        ops.append(lambda im: ImageEnhance.Contrast(im).enhance(random.uniform(1.05, 1.5)))
    if do_brightness:
        ops.append(lambda im: ImageEnhance.Brightness(im).enhance(random.uniform(0.9, 1.2)))
    if do_sharpness:
        ops.append(lambda im: ImageEnhance.Sharpness(im).enhance(random.uniform(1.0, 2.0)))
    if do_noise and np is not None:
        def noise(im):
            arr = np.array(im).astype(np.float32)
            sigma = random.uniform(2.0, 8.0)
            noise = np.random.normal(0, sigma, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        ops.append(noise)

    if not ops:
        return img

    op = random.choice(ops)
    return op(img)

# -------- GUI --------
class PreprocessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IQ-OTH/NCCD Preprocessing")
        self.root.geometry("980x640")

        self.base_dir = tk.StringVar()
        self.out_dir = tk.StringVar()

        self.target_w = tk.IntVar(value=224)
        self.target_h = tk.IntVar(value=224)
        self.target_count = tk.IntVar(value=600)
        self.preview_use_aug = tk.BooleanVar(value=True)

        # Aug flags
        self.rot_deg = tk.IntVar(value=25)
        self.aug_flip = tk.BooleanVar(value=True)
        self.aug_rot = tk.BooleanVar(value=True)
        self.aug_contrast = tk.BooleanVar(value=True)
        self.aug_brightness = tk.BooleanVar(value=True)
        self.aug_sharp = tk.BooleanVar(value=True)
        self.aug_noise = tk.BooleanVar(value=False)

        self._build_ui()

    def _build_ui(self):
        # ----- Paths frame -----
        frm_paths = ttk.LabelFrame(self.root, text="Pastas")
        frm_paths.pack(fill="x", padx=10, pady=10)

        ttk.Label(frm_paths, text="Base Dir (dataset original):").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(frm_paths, textvariable=self.base_dir, width=80).grid(row=0, column=1, padx=8)
        ttk.Button(frm_paths, text="Selecionar…", command=self.select_base).grid(row=0, column=2, padx=8)

        ttk.Label(frm_paths, text="Output Dir (dataset processado):").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(frm_paths, textvariable=self.out_dir, width=80).grid(row=1, column=1, padx=8)
        ttk.Button(frm_paths, text="Selecionar…", command=self.select_out).grid(row=1, column=2, padx=8)

        # ----- Config frame -----
        frm_cfg = ttk.LabelFrame(self.root, text="Configurações")
        frm_cfg.pack(fill="x", padx=10, pady=10)

        # Resize
        ttk.Label(frm_cfg, text="Resize (W×H):").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(frm_cfg, textvariable=self.target_w, width=8).grid(row=0, column=1, padx=4)
        ttk.Entry(frm_cfg, textvariable=self.target_h, width=8).grid(row=0, column=2, padx=4)

        # Balance
        ttk.Label(frm_cfg, text="Target por classe (balanceamento):").grid(row=0, column=3, sticky="w", padx=16)
        ttk.Entry(frm_cfg, textvariable=self.target_count, width=8).grid(row=0, column=4)

        # Aug options
        ttk.Label(frm_cfg, text="Augmentations:").grid(row=1, column=0, sticky="w", padx=8)

        ttk.Checkbutton(frm_cfg, text="Preview com Augmentations", variable=self.preview_use_aug).grid(row=3, column=1, sticky="w", pady=(4, 0))
        ttk.Checkbutton(frm_cfg, text="Horizontal Flip", variable=self.aug_flip).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(frm_cfg, text="Rotação", variable=self.aug_rot).grid(row=1, column=2, sticky="w")
        ttk.Label(frm_cfg, text="± graus").grid(row=1, column=3, sticky="e")
        ttk.Entry(frm_cfg, textvariable=self.rot_deg, width=6).grid(row=1, column=4, sticky="w", padx=4)

        ttk.Checkbutton(frm_cfg, text="Contraste", variable=self.aug_contrast).grid(row=2, column=1, sticky="w")
        ttk.Checkbutton(frm_cfg, text="Brilho", variable=self.aug_brightness).grid(row=2, column=2, sticky="w")
        ttk.Checkbutton(frm_cfg, text="Nitidez", variable=self.aug_sharp).grid(row=2, column=3, sticky="w")
        ttk.Checkbutton(frm_cfg, text="Ruído (leve)", variable=self.aug_noise).grid(row=2, column=4, sticky="w")

        # ----- Buttons frame -----
        frm_btns = ttk.Frame(self.root)
        frm_btns.pack(fill="x", padx=10, pady=5)

        ttk.Button(frm_btns, text="Contar por classe", command=self.count_images).pack(side="left", padx=6)
        ttk.Button(frm_btns, text="Redimensionar (copiar + resize)", command=self.resize_only).pack(side="left", padx=6)
        ttk.Button(frm_btns, text="Augment + Balancear", command=self.augment_and_balance).pack(side="left", padx=6)
        ttk.Button(frm_btns, text="Preview Aug", command=self.preview_aug).pack(side="left", padx=6)
        ttk.Button(frm_btns, text="Abrir Output", command=self.open_out_dir).pack(side="left", padx=6)
        ttk.Button(frm_btns, text="Selecionar imagem e Preview", command=self.preview_selected).pack(side="left", padx=6)

        # ----- Log frame -----
        frm_log = ttk.LabelFrame(self.root, text="Log")
        frm_log.pack(fill="both", expand=True, padx=10, pady=10)

        self.txt_log = tk.Text(frm_log, height=16)
        self.txt_log.pack(fill="both", expand=True)

    # -------- Actions --------
    def log(self, msg: str):
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.root.update_idletasks()

    def select_base(self):
        d = filedialog.askdirectory(title="Selecione a pasta base do dataset")
        if d:
            self.base_dir.set(d)

    def select_out(self):
        d = filedialog.askdirectory(title="Selecione a pasta de saída (ou crie uma nova e selecione)")
        if d:
            self.out_dir.set(d)

    def open_out_dir(self):
        p = self.out_dir.get().strip()
        if not p or not os.path.isdir(p):
            messagebox.showinfo("Info", "Output Dir inválido.")
            return
        # Tenta abrir no SO (funciona em Win/Mac/Linux)
        if os.name == "nt":
            os.startfile(p)
        elif os.uname().sysname == "Darwin":
            os.system(f'open "{p}"')
        else:
            os.system(f'xdg-open "{p}"')

    def _validate_paths(self):
        base = self.base_dir.get().strip()
        out = self.out_dir.get().strip()
        if not base or not os.path.isdir(base):
            messagebox.showerror("Erro", "Base Dir inválido.")
            return None, None, None
        if not out:
            messagebox.showerror("Erro", "Output Dir inválido.")
            return None, None, None

        cls_map = find_class_folders(base)
        if not cls_map:
            messagebox.showerror("Erro", "Não encontrei pastas de classe (Benign/Malignant/Normal). Verifique os nomes.")
            return None, None, None
        return base, out, cls_map

    def count_images(self):
        try:
            base, _, cls_map = self._validate_paths()
            if base is None:
                return
            self.log("== Contagem de imagens ==")
            for cname in STANDARD_CLASSES:
                if cname in cls_map:
                    cnt = len(list_images(cls_map[cname]))
                    self.log(f"{cname}: {cnt}")
                else:
                    self.log(f"{cname}: (não encontrada)")
            self.log("— fim —")
        except Exception as e:
            self.log("Erro na contagem:\n" + traceback.format_exc())

    def resize_only(self):
        try:
            base, out, cls_map = self._validate_paths()
            if base is None:
                return

            target = (self.target_w.get(), self.target_h.get())
            self.log(f"== Redimensionar para {target[0]}x{target[1]} ==")

            for cname in STANDARD_CLASSES:
                if cname not in cls_map:
                    continue
                src = cls_map[cname]
                dst = os.path.join(out, cname)
                ensure_dir(dst)

                imgs = list_images(src)
                self.log(f"{cname}: copiando + redimensionando {len(imgs)} imagens...")
                for img_name in imgs:
                    try:
                        src_path = os.path.join(src, img_name)
                        img = safe_open_rgb(src_path, target_size=target)
                        img.save(os.path.join(dst, img_name))
                    except Exception as ex:
                        self.log(f"[WARN] Falha em {img_name}: {ex}")

            self.log("— redimensionamento concluído —")
        except Exception as e:
            self.log("Erro no redimensionamento:\n" + traceback.format_exc())

    def augment_and_balance(self):
        try:
            base, out, cls_map = self._validate_paths()
            if base is None:
                return

            target = (self.target_w.get(), self.target_h.get())
            tgt_count = max(1, int(self.target_count.get()))

            aug_flags = dict(
                rot_deg=self.rot_deg.get(),
                do_hflip=self.aug_flip.get(),
                do_rot=self.aug_rot.get(),
                do_contrast=self.aug_contrast.get(),
                do_brightness=self.aug_brightness.get(),
                do_sharpness=self.aug_sharp.get(),
                do_noise=self.aug_noise.get(),
            )

            self.log(f"== Augment + Balancear para {tgt_count} por classe (size={target[0]}x{target[1]}) ==")

            for cname in STANDARD_CLASSES:
                if cname not in cls_map:
                    self.log(f"{cname}: (não encontrada) — pulando")
                    continue

                src = cls_map[cname]
                dst = os.path.join(out, cname)
                if os.path.isdir(dst):
                    # para evitar reuso confuso, limpamos e recriamos
                    shutil.rmtree(dst)
                ensure_dir(dst)

                imgs = list_images(src)
                self.log(f"{cname}: base com {len(imgs)} imagens.")
                # primeiro copia + resize todas
                for img_name in imgs:
                    try:
                        src_path = os.path.join(src, img_name)
                        img = safe_open_rgb(src_path, target_size=target)
                        img.save(os.path.join(dst, img_name))
                    except Exception as ex:
                        self.log(f"[WARN] Falha copiando {img_name}: {ex}")

                # depois gera o extra, se necessário
                extra_needed = tgt_count - len(imgs)
                if extra_needed <= 0:
                    self.log(f"{cname}: já possui >= {tgt_count}. Sem oversampling.")
                    continue

                self.log(f"{cname}: gerando {extra_needed} amostras sintéticas...")
                for i in range(extra_needed):
                    try:
                        seed_name = random.choice(imgs)
                        spath = os.path.join(src, seed_name)
                        img = safe_open_rgb(spath, target_size=target)
                        aug_img = augment_random(img, **aug_flags)
                        save_name = f"aug_{i:05d}_{seed_name}"
                        aug_img.save(os.path.join(dst, save_name))
                    except Exception as ex:
                        self.log(f"[WARN] Aug falhou em {seed_name}: {ex}")

            self.log("— augmentation + balanceamento concluídos —")
        except Exception as e:
            self.log("Erro no augmentation/balanceamento:\n" + traceback.format_exc())
    
    def preview_selected(self):
        try:
            base, _, cls_map = self._validate_paths()
            if base is None:
                return

            # escolhe um arquivo manualmente
            img_path = filedialog.askopenfilename(
                title="Selecione uma imagem para preview",
                filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
            )
            if not img_path:
                return

            # confirma que está dentro de alguma pasta de classe (opcional, mas útil)
            if not any(os.path.commonpath([img_path, cls_map[c]]) == cls_map[c] for c in cls_map):
                if not messagebox.askyesno(
                    "Aviso",
                    "A imagem selecionada não está dentro das pastas de classe detectadas.\n"
                    "Deseja continuar mesmo assim?"
                ):
                    return

            target = (self.target_w.get(), self.target_h.get())
            apply_aug = self.preview_use_aug.get()

            aug_flags = dict(
                rot_deg=self.rot_deg.get(),
                do_hflip=self.aug_flip.get(),
                do_rot=self.aug_rot.get(),
                do_contrast=self.aug_contrast.get(),
                do_brightness=self.aug_brightness.get(),
                do_sharpness=self.aug_sharp.get(),
                do_noise=self.aug_noise.get(),
            )

            # carrega original (sem resize/augment) só para exibir
            img_orig = Image.open(img_path).convert("RGB")
            # processado: resize + (opcional) augment
            img_proc = img_orig.resize(target, Image.BILINEAR)
            if apply_aug:
                img_proc = augment_random(img_proc, **aug_flags)

            # janela de preview
            win = tk.Toplevel(self.root)
            win.title(f"Preview selecionado: {os.path.basename(img_path)}")
            frm = ttk.Frame(win, padding=10)
            frm.pack(fill="both", expand=True)

            tk.Label(frm, text="Original").grid(row=0, column=0, padx=8, pady=6)
            tk.Label(frm, text=f"Processado ({target[0]}x{target[1]}{', com Aug' if apply_aug else ', sem Aug'})").grid(row=0, column=1, padx=8, pady=6)

            # para caber bem, redimensiona a visualização se a imagem original for muito grande
            def to_photo(im: Image.Image, max_side=512):
                w, h = im.size
                scale = min(max_side / max(w, h), 1.0)
                if scale < 1.0:
                    im = im.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
                return ImageTk.PhotoImage(im)

            ph1 = to_photo(img_orig)
            ph2 = to_photo(img_proc)

            lbl1 = tk.Label(frm, image=ph1)
            lbl2 = tk.Label(frm, image=ph2)
            lbl1.image = ph1
            lbl2.image = ph2
            lbl1.grid(row=1, column=0, padx=8, pady=8)
            lbl2.grid(row=1, column=1, padx=8, pady=8)

        except Exception as e:
            self.log("Erro no preview selecionado:\n" + traceback.format_exc())


    def preview_aug(self):
        try:
            base, _, cls_map = self._validate_paths()
            if base is None:
                return

            # pega uma classe existente com ao menos 1 imagem
            chosen_cls = None
            for cname in STANDARD_CLASSES:
                if cname in cls_map and len(list_images(cls_map[cname])) > 0:
                    chosen_cls = cname
                    break
            if not chosen_cls:
                messagebox.showwarning("Aviso", "Nenhuma imagem encontrada para preview.")
                return

            src = cls_map[chosen_cls]
            imgs = list_images(src)
            seed_name = random.choice(imgs)
            spath = os.path.join(src, seed_name)
            target = (self.target_w.get(), self.target_h.get())

            aug_flags = dict(
                rot_deg=self.rot_deg.get(),
                do_hflip=self.aug_flip.get(),
                do_rot=self.aug_rot.get(),
                do_contrast=self.aug_contrast.get(),
                do_brightness=self.aug_brightness.get(),
                do_sharpness=self.aug_sharp.get(),
                do_noise=self.aug_noise.get(),
            )

            img = safe_open_rgb(spath, target_size=target)
            aug_img = augment_random(img, **aug_flags)

            # mostra antes/depois
            win = tk.Toplevel(self.root)
            win.title(f"Preview: {chosen_cls} / {seed_name}")
            frm = ttk.Frame(win, padding=10)
            frm.pack(fill="both", expand=True)

            tk.Label(frm, text="Original").grid(row=0, column=0, padx=8, pady=6)
            tk.Label(frm, text="Augmentada").grid(row=0, column=1, padx=8, pady=6)

            ph1 = ImageTk.PhotoImage(img)
            ph2 = ImageTk.PhotoImage(aug_img)
            lbl1 = tk.Label(frm, image=ph1)
            lbl2 = tk.Label(frm, image=ph2)
            lbl1.image = ph1
            lbl2.image = ph2
            lbl1.grid(row=1, column=0, padx=8, pady=8)
            lbl2.grid(row=1, column=1, padx=8, pady=8)

        except Exception as e:
            self.log("Erro no preview:\n" + traceback.format_exc())


if __name__ == "__main__":
    root = tk.Tk()
    app = PreprocessingGUI(root)
    root.mainloop()
