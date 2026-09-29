"""Extract frozen image/text embeddings exactly as notebooks 04/05/07/08 did.

Differences from the notebooks are limited to what this machine forces and are recorded in
embeddings/extraction_report.json:
* DINOv3-L: the gated `facebook/dinov3-vitl16-pretrain-lvd1689m` repo is not accessible with the
  available token, so the timm port of the same ViT-L/16 LVD-1689M weights is used, with the
  DINOv3 image processor re-implemented op for op (see `Dinov3Preprocess`; checked bit-exact
  against transformers' DINOv3ViTImageProcessor at load time) and the post-final-LayerNorm CLS
  token as the embedding (= HF `pooler_output`).
* The thesis ran transformers 4.5x (DINOv3 via HF needs >= 4.56); this machine has 5.x, whose
  defaults changed in two ways that alter embeddings, so the 4.x behaviour is requested
  explicitly: (a) `from_pretrained` now loads the checkpoint dtype ("auto"), which is float16 for
  multilingual-e5-large-instruct -> `dtype=torch.float32`; (b) AutoImageProcessor now defaults to
  the torchvision backend, while 4.x loaded the slow PIL BitImageProcessor saved with
  facebook/dinov2-large -> `backend="pil"`.
* Preprocessing callables are module-level classes (Windows DataLoader workers must pickle them);
  their arithmetic is the same as the closures in `crm.encoders.image`.
* Images are opened with `Image.open(p).convert('RGB')` (no EXIF transpose), as in the notebooks.
  Unreadable/missing images get an all-zero embedding row, as in the notebooks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import functional as tvF

from common import DEFAULT_ARTIFACTS, import_crm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BATCH = {"dinov3_large": 64, "dinov2_large": 64, "eva02_large": 16, "mE5_large": 32}
EMB_DIM = 1024  # every encoder of the replication (crm IMAGE_ENCODERS / TEXT_ENCODERS dim)

# preprocessor_config.json of facebook/dinov3-vitl16-pretrain-lvd1689m (DINOv3ViTImageProcessorFast,
# transformers 4.56; identical copy in the ungated onnx-community/dinov3-vitl16-pretrain-lvd1689m-ONNX)
DINOV3_PROCESSOR_CONFIG = {
    "do_resize": True, "size": {"height": 224, "width": 224}, "resample": 2,  # PIL BILINEAR
    "do_rescale": True, "rescale_factor": 0.00392156862745098,
    "do_normalize": True, "image_mean": list(IMAGENET_MEAN), "image_std": list(IMAGENET_STD),
}


# ----------------------------------------------------------------------------- preprocessors
class Dinov3Preprocess:
    """Op-for-op copy of transformers' DINOv3ViTImageProcessor(Fast) with DINOV3_PROCESSOR_CONFIG.

    DINOv3 overrides the processor order to rescale -> resize -> normalize: the uint8 image is
    first multiplied by 1/255 (float32) and the *float* tensor is resized to 224x224 (no crop,
    aspect ratio not kept) with torchvision bilinear + antialias. A Pillow uint8 resize is not
    equivalent (it rounds to uint8 and uses a different kernel; max |diff| ~0.017 after
    normalisation on the thesis images).
    """

    def __init__(self, size: int = 224, rescale_factor: float = DINOV3_PROCESSOR_CONFIG["rescale_factor"]):
        self.size = size
        self.rescale_factor = rescale_factor
        self.mean = list(IMAGENET_MEAN)
        self.std = list(IMAGENET_STD)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        x = tvF.pil_to_tensor(img).unsqueeze(0)  # (1, 3, H, W) uint8, batched like the processor
        x = x * self.rescale_factor  # float32
        x = tvF.resize(x, [self.size, self.size], interpolation=tvF.InterpolationMode.BILINEAR,
                       antialias=True)
        return tvF.normalize(x, self.mean, self.std)[0]


def check_dinov3_preprocess(pre: Dinov3Preprocess) -> str:
    """Assert bit-equality with the installed transformers DINOv3 processor (guards version drift)."""
    from transformers import DINOv3ViTImageProcessor

    ref = DINOv3ViTImageProcessor(**DINOV3_PROCESSOR_CONFIG)
    rng = np.random.default_rng(0)
    for h, w in ((480, 640), (1280, 720), (97, 301), (224, 224), (1707, 1280)):
        img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")
        a = ref(images=img, return_tensors="pt")["pixel_values"][0]
        b = pre(img)
        if a.shape != b.shape or not torch.equal(a, b):
            raise RuntimeError(f"Dinov3Preprocess differs from DINOv3ViTImageProcessor on a {w}x{h} image "
                               f"(max |diff| {float((a - b).abs().max()) if a.shape == b.shape else 'shape'})")
    return f"bit-identical to transformers {__import__('transformers').__version__} DINOv3ViTImageProcessor"


class HFProcessor:
    """Picklable wrapper around an HF AutoImageProcessor (same call as crm.encoders.image._hf_loader)."""

    def __init__(self, hf_id: str, backend: str | None = None):
        from transformers import AutoImageProcessor

        kwargs = {} if backend is None else {"backend": backend}
        self.processor = AutoImageProcessor.from_pretrained(hf_id, **kwargs)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]


class CLSFeatures(torch.nn.Module):
    """timm DINOv3 port: forward_features ends with the final LayerNorm; take the CLS token."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_features(x)[:, 0]


def load_image_encoder(name: str, device: str):
    crm = import_crm()
    from crm.encoders.image import IMAGE_ENCODERS

    if name == "dinov3_large":
        import timm

        m = timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0)
        # index 0 is the CLS token (timm: [cls, 4 registers, patches], same order as HF) and
        # forward_features applies the final LayerNorm (use_fc_norm=False -> post norm).
        assert m.num_prefix_tokens == 5 and not isinstance(m.norm, torch.nn.Identity)
        pre = Dinov3Preprocess(224)
        return CLSFeatures(m).eval().to(device), pre, {
            "source": "timm/vit_large_patch16_dinov3.lvd1689m (port of facebook/dinov3-vitl16-pretrain-lvd1689m)",
            "preprocess": "DINOv3ViTImageProcessor re-implementation: x/255 -> resize 224x224 bilinear "
                          "antialias (torchvision, float) -> ImageNet mean/std; no crop",
            "preprocess_check": check_dinov3_preprocess(pre),
            "pool": "post-final-LayerNorm CLS token (= HF pooler_output)"}
    spec = IMAGE_ENCODERS[name]
    if spec.hf_id.startswith("timm/"):
        import timm
        from timm.data import create_transform, resolve_data_config

        model = timm.create_model(spec.hf_id.removeprefix("timm/"), pretrained=True, num_classes=0).eval().to(device)
        cfg = resolve_data_config({}, model=model)
        cfg["input_size"] = (3, spec.image_size, spec.image_size)
        return model, create_transform(**cfg, is_training=False), {
            "source": spec.hf_id, "preprocess": f"timm resolve_data_config, input {spec.image_size}",
            "pool": "timm model(x) with num_classes=0"}
    from transformers import AutoModel

    # transformers 4.x defaults: fp32 weights and the slow (PIL) processor saved with the checkpoint
    model = AutoModel.from_pretrained(spec.hf_id, dtype=torch.float32).eval().to(device)
    proc = HFProcessor(spec.hf_id, backend="pil")
    return model, proc, {"source": spec.hf_id,
                         "preprocess": f"AutoImageProcessor {type(proc.processor).__name__} (PIL backend)",
                         "pool": "pooler_output", "weights_dtype": str(next(model.parameters()).dtype)}


# ----------------------------------------------------------------------------- data
class ImageRows(Dataset):
    def __init__(self, paths: list[Path], preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return idx, self.preprocess(img), None
        except Exception as e:  # noqa: BLE001 - mirror the notebooks: skip unreadable images
            return idx, None, str(e)[:120]


def collate(batch):
    valid = [(i, t) for i, t, e in batch if t is not None]
    errs = [(i, e) for i, t, e in batch if e is not None]
    if not valid:
        return [], None, errs
    idxs, tensors = zip(*valid)
    return list(idxs), torch.stack(list(tensors)), errs


def pool(out):
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]
    if isinstance(out, torch.Tensor):
        return out
    raise ValueError(f"Unknown output: {type(out)}")


def save_npy(path: Path, arr: np.ndarray) -> None:
    """Write atomically: an interrupted save must not leave a truncated cache that looks complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def cached(out_path: Path, n_rows: int, entry: dict, fingerprint: str) -> bool:
    """Reuse a cache only if it has the right rows AND was built from this metadata_clean.csv."""
    if not out_path.exists() or entry.get("metadata_clean_sha256") != fingerprint:
        return False
    try:
        return np.load(out_path, mmap_mode="r").shape == (n_rows, EMB_DIM)
    except (ValueError, OSError):
        return False


def extract_image(name: str, df: pd.DataFrame, base: Path, out_dir: Path, device: str, workers: int) -> dict:
    out_path = out_dir / "image" / f"{name}.npy"
    model, preprocess, meta = load_image_encoder(name, device)
    ds = ImageRows([base / p for p in df["gambar"]], preprocess)
    loader = DataLoader(ds, batch_size=BATCH[name], num_workers=workers, collate_fn=collate,
                        pin_memory=device == "cuda", persistent_workers=workers > 0,
                        prefetch_factor=4 if workers > 0 else None)
    emb = np.zeros((len(df), EMB_DIM), dtype=np.float32)
    failures = []
    t0 = time.time()
    with torch.inference_mode():
        for n, (idxs, tensors, errs) in enumerate(loader):
            failures.extend(errs)
            if tensors is None:
                continue
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                feats = pool(model(tensors.to(device, non_blocking=True))).float().cpu().numpy()
            if feats.shape[1] != EMB_DIM:
                raise RuntimeError(f"{name}: embedding dim {feats.shape[1]} != {EMB_DIM}")
            emb[idxs] = feats
            if n % 50 == 0:
                done = min((n + 1) * BATCH[name], len(df))
                print(f"  {name}: {done:,}/{len(df):,} ({done / max(time.time() - t0, 1e-6):.0f} img/s)", flush=True)
    del loader, model
    torch.cuda.empty_cache()
    if len(failures) == len(df):
        raise RuntimeError(f"{name}: every image failed to load (e.g. {failures[:3]}); wrong --artifacts?")
    save_npy(out_path, emb)
    return {**meta, "shape": list(emb.shape), "failed_images_zero_rows": len(failures),
            "failed_examples": failures[:10],
            "zero_rows": int((np.abs(emb).sum(1) == 0).sum()), "seconds": round(time.time() - t0, 1)}


def extract_text(name: str, df: pd.DataFrame, out_dir: Path, device: str) -> dict:
    out_path = out_dir / "text" / f"{name}.npy"
    import_crm()
    from transformers import AutoModel, AutoTokenizer

    from crm.encoders.text import TEXT_ENCODERS, encode_batch

    spec = TEXT_ENCODERS[name]
    url_re = re.compile(r"https?://\S+|www\.\S+")  # notebook 05 cleaning
    ws_re = re.compile(r"\s+")
    texts = [ws_re.sub(" ", url_re.sub("", str(t).strip())).strip() for t in df["laporan"]]
    # = crm.encoders.text.load_encoder, but with the transformers 4.x default dtype (fp32): 5.x
    # would load this checkpoint in float16 because its config says torch_dtype=float16.
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModel.from_pretrained(spec.hf_id, dtype=torch.float32).eval().to(device)
    emb = np.zeros((len(df), spec.dim), dtype=np.float32)
    t0 = time.time()
    bs = BATCH[name]
    for start in range(0, len(texts), bs):
        emb[start:start + bs] = encode_batch(model, tok, texts[start:start + bs], spec, device, use_amp=True)
        if (start // bs) % 100 == 0:
            print(f"  {name}: {min(start + bs, len(texts)):,}/{len(texts):,}", flush=True)
    save_npy(out_path, emb)
    info = {"source": spec.hf_id, "prefix": spec.prefix, "pool": spec.pool, "max_length": spec.max_length,
            "tokenizer": type(tok).__name__, "weights_dtype": str(next(model.parameters()).dtype),
            "shape": list(emb.shape), "seconds": round(time.time() - t0, 1)}
    del model
    torch.cuda.empty_cache()
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    ap.add_argument("--encoders", nargs="+", default=["mE5_large", "dinov3_large", "dinov2_large", "eva02_large"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    art = Path(args.artifacts)
    base = art / "crm_jakarta"
    meta_path = base / "metadata_clean.csv"
    fingerprint = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    df = pd.read_csv(meta_path, low_memory=False).reset_index(drop=True)
    out_dir = art / "embeddings"
    report_path = out_dir / "extraction_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    print(f"device={device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}), rows={len(df):,}")
    for name in args.encoders:
        print(f"=== {name}", flush=True)
        modality = "text" if name == "mE5_large" else "image"
        if cached(out_dir / modality / f"{name}.npy", len(df), report.get(name, {}), fingerprint):
            print("    [skip] cached for this metadata_clean.csv", flush=True)
            continue
        info = extract_text(name, df, out_dir, device) if modality == "text" else \
            extract_image(name, df, base, out_dir, device, args.workers)
        report[name] = {**info, "metadata_clean_sha256": fingerprint, "rows": len(df),
                        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                        "timm": __import__("timm").__version__, "device": device}
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"    {json.dumps(info)}", flush=True)  # ASCII-escaped: safe on a cp1252 console


if __name__ == "__main__":
    main()
