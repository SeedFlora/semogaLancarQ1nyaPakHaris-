"""Download the four encoder checkpoints (safetensors only) into HF_HUB_CACHE, with retries."""

import time

from huggingface_hub import snapshot_download

REPOS = ["intfloat/multilingual-e5-large-instruct", "timm/vit_large_patch16_dinov3.lvd1689m",
         "facebook/dinov2-large", "timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"]

for rid in REPOS:
    for attempt in range(10):
        try:
            t = time.time()
            p = snapshot_download(rid, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
            print("OK", rid, p, f"{time.time() - t:.0f}s", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print("RETRY", rid, attempt, type(e).__name__, str(e)[:100], flush=True)
            time.sleep(10)
    else:
        print("FAILED", rid, flush=True)
print("ALL_DONE", flush=True)
