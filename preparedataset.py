"""
prepare_dataset.py  --  Persiapan dataset proyek "Klasifikasi Sampah Rumah Tangga"
Mata Kuliah Computer Vision | Anggota 1 (Data Specialist)

Yang dilakukan skrip ini (semua deterministik, seed = 42):
  1. Mengunduh TrashNet (dataset-resized.zip, 512x384) dari GitHub dan mengekstraknya.
  2. Menyeleksi kelas: cardboard + paper -> "kertas_karton", plastic -> "plastik",
     metal -> "logam". Kelas glass dan trash tidak dipakai.
  3. Memeriksa integritas (file terbaca, RGB) dan duplikat byte-identik (MD5).
  4. Mengelompokkan citra near-duplicate (foto objek yang sama dari sudut sedikit
     berbeda) dengan dHash 256-bit, ambang jarak Hamming <= 12, hanya dalam kelas yang sama.
  5. Membagi data 70/15/15 (latih/validasi/uji) secara stratified DAN berbasis grup,
     sehingga near-duplicate tidak pernah terpisah antara latih dan uji.
  6. Menyalin file ke struktur folder dataset/{train,val,test}/{kelas}/ dan menyimpan
     splits.csv serta class_weights.json.

Kebutuhan: pip install numpy pandas pillow scikit-learn scipy
Pemakaian :
  # sudah punya dataset (folder berisi subfolder cardboard, glass, metal, paper, plastic, trash):
  python prepare_dataset.py --data-dir "path/ke/folder_dataset" --out dataset_final
  # belum punya dataset (unduh otomatis dari GitHub):
  python prepare_dataset.py --out dataset_final
"""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

URL = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
SEED = 42
LABEL_MAP = {"plastic": "plastik", "paper": "kertas_karton",
             "cardboard": "kertas_karton", "metal": "logam"}
HASH_T = 12          # ambang jarak Hamming (dari 256 bit) untuk near-duplicate
N_FOLDS = 20         # 14 fold latih (70%), 3 fold validasi (15%), 3 fold uji (15%)


EXPECTED = {"cardboard": 403, "paper": 594, "plastic": 482, "metal": 410}  # TrashNet resmi
EXTS = {".jpg", ".jpeg", ".png"}


def find_class_root(base: Path, max_depth: int = 3):
    """Cari folder yang berisi subfolder kelas (cardboard/metal/paper/plastic), tidak peka huruf besar-kecil."""
    need = set(EXPECTED)
    cands = [base] + [p for p in base.rglob("*") if p.is_dir() and len(p.relative_to(base).parts) <= max_depth]
    for c in cands:
        names = {d.name.lower(): d for d in c.iterdir() if d.is_dir()}
        if need <= set(names):
            return c, {k: names[k] for k in need}
    return None, None


def download_and_extract(raw: Path) -> Path:
    raw.mkdir(parents=True, exist_ok=True)
    root, _ = find_class_root(raw)
    if root is not None:
        return root  # sudah ada, tidak perlu unduh
    zpath = raw / "dataset-resized.zip"
    if not zpath.exists():
        print("Mengunduh TrashNet ...")
        urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(raw)
    return find_class_root(raw)[0]


def dhash256(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L").resize((17, 16), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def main(out: Path, raw: Path, data_dir: Path = None):
    if data_dir is not None:
        root, class_dirs = find_class_root(data_dir)
        if root is None:
            raise SystemExit(f"Tidak menemukan subfolder cardboard/metal/paper/plastic di {data_dir}")
    else:
        root = download_and_extract(raw)
        class_dirs = find_class_root(root)[1]
    print("Folder dataset:", root.resolve())

    # --- 2. seleksi kelas dan pemetaan label
    rows = []
    for kelas_asal, label in LABEL_MAP.items():
        for f in sorted(class_dirs[kelas_asal].iterdir()):
            if f.suffix.lower() in EXTS:
                rows.append((f"{kelas_asal}/{f.name}", kelas_asal, label, str(f)))
    df = (pd.DataFrame(rows, columns=["path", "kelas_asal", "label", "src"])
          .sort_values("path").reset_index(drop=True))
    print("Jumlah citra terpilih:", len(df))
    print(df.label.value_counts().to_string())
    got = df.kelas_asal.value_counts().to_dict()
    if got != EXPECTED:
        print(f"PERINGATAN: jumlah per kelas {got} berbeda dari TrashNet resmi {EXPECTED}. "
              "Angka di proposal (1.889 citra; 1.321/284/284) mungkin perlu disesuaikan.")

    # --- 3. integritas + duplikat byte-identik
    md5 = {}
    for p, src in zip(df.path, df.src):
        with Image.open(src) as im:
            assert im.mode == "RGB", f"{p} bukan RGB (mode {im.mode}); konversi dulu ke RGB"
        md5.setdefault(hashlib.md5(Path(src).read_bytes()).hexdigest(), []).append(p)
    dups = [v for v in md5.values() if len(v) > 1]
    print("Duplikat byte-identik pada kelas terpilih:", len(dups))
    assert not dups, dups

    # --- 4. grup near-duplicate
    H = np.stack([dhash256(Path(s)) for s in df.src])
    D = (H[:, None, :] != H[None, :, :]).sum(-1)
    lab = pd.factorize(df.label)[0]
    near = (D <= HASH_T) & (lab[:, None] == lab[None, :])
    n_grup, grup = connected_components(csr_matrix(near), directed=False)
    sz = np.bincount(grup)
    print(f"Grup: {n_grup} | grup berisi >1 citra: {(sz > 1).sum()} "
          f"({sz[sz > 1].sum()} citra) | grup terbesar: {sz.max()}")
    df["grup"] = grup

    # --- 5. split stratified + berbasis grup (70/15/15)
    y = pd.factorize(df.kelas_asal)[0]  # stratifikasi pada 4 kelas asal
    fold = np.full(len(df), -1)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, te) in enumerate(sgkf.split(df, y, groups=grup)):
        fold[te] = k
    perm = np.random.default_rng(SEED).permutation(N_FOLDS)
    role = {f: ("train" if r < 14 else "val" if r < 17 else "test") for r, f in enumerate(perm)}
    df["split"] = [role[f] for f in fold]
    print(pd.crosstab(df.label, df.split, margins=True).to_string())

    # --- 6. salin file + simpan metadata
    if out.exists():
        shutil.rmtree(out)
    for _, r in df.iterrows():
        dst = out / r.split / r.label
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r.src, dst / f"{r.kelas_asal}_{Path(r.path).name}")
    df.drop(columns=["src"]).to_csv(out / "splits.csv", index=False)

    tr = df[df.split == "train"].label.value_counts()
    cw = {k: float(len(df[df.split == 'train']) / (len(tr) * v)) for k, v in tr.items()}
    json.dump(cw, open(out / "class_weights.json", "w"), indent=2)
    print("Bobot kelas (N / (K * n_kelas)) dari data latih:", {k: round(v, 3) for k, v in cw.items()})
    print("Selesai ->", out.resolve())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_final", type=Path)
    ap.add_argument("--data-dir", default=None, type=Path,
                    help="folder dataset yang sudah kamu punya (berisi subfolder kelas)")
    ap.add_argument("--keep-raw", default="trashnet_raw", type=Path,
                    help="folder unduhan jika --data-dir tidak diberikan")
    a = ap.parse_args()
    main(a.out, a.keep_raw, a.data_dir)