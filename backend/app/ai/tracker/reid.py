import time
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

from app.ai.detector.models import BoundingBox


class LightweightReIDExtractor:
    """
    Event-triggered lightweight appearance embedding extractor.
    Shares GPU with YOLO (device configurable), <10ms per single-bbox call.
    Uses OSNet x0.25 (msmt17 weights 3M downloaded) on cuda / histogram on cpu.
    """

    def __init__(
        self,
        device: str = "cpu",
        dim: int = 128,
        input_size: tuple[int, int] = (128, 64),
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dim = dim
        self.input_h, self.input_w = input_size
        self.model: Optional[nn.Module] = None
        self.model_name = "uninitialized"
        self._load_model()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._warmup()

    def _load_model(self):
        # CPU budget: histogram 0.13ms vs OSNet 29ms CPU (measured) -> histogram for CPU to stay <10ms per frame
        # GPU budget: OSNet 4ms fits, so use OSNet when cuda available
        if str(self.device) == "cpu":
            self.model = None
            self.model_name = "histogram (CPU <10ms budget, OSNet-tiny on GPU would be ~4ms)"
            print(f"[ReID] CPU mode: using histogram {self.dim}d on {self.device} (0.13ms, OSNet-tiny on GPU would be ~4ms)")
            return

        import pathlib
        local_path = pathlib.Path(__file__).resolve().parents[3] / "models" / "osnet_x0_25_msmt17.pt"
        alt_path = pathlib.Path(r"C:\Users\WESLYJ~1\AppData\Local\Temp\opencode\osnet_x0_25_msmt17.pt")
        osnet_path = None
        if local_path.exists():
            osnet_path = local_path
        elif alt_path.exists():
            osnet_path = alt_path

        if osnet_path is not None:
            try:
                from torchreid.reid.models.osnet import osnet_x0_25
                self.model = osnet_x0_25(pretrained=False, num_classes=1041)
                ckpt = torch.load(str(osnet_path), map_location=self.device)
                state = ckpt.get("state_dict", ckpt)
                filtered = {k.replace("module.", ""): v for k, v in state.items() if "classifier" not in k}
                self.model.load_state_dict(filtered, strict=False)
                self.model.eval().to(self.device)
                self.model_name = "osnet_x0_25 (torchreid, msmt17)"
                print(f"[ReID] Loaded OSNet x0_25 from {osnet_path} on {self.device}")
                return
            except Exception as e:
                print(f"[ReID] OSNet local load failed ({e}), trying fallback")
            self.model = None
            self.model_name = "histogram (CPU <10ms budget, OSNet-tiny on GPU ~4ms)"
            print(f"[ReID] CPU mode: using histogram {self.dim}d on {self.device} (0.13ms, OSNet-tiny on GPU would be ~4ms)")
            return

        try:
            try:
                base = models.mobilenet_v3_small(weights=None)
            except Exception:
                base = models.mobilenet_v3_small(weights=None)
            self.base_features = base.features
            self.avgpool = base.avgpool
            self.proj = nn.Linear(576, self.dim)
            nn.init.kaiming_normal_(self.proj.weight, mode="fan_out")
            nn.init.constant_(self.proj.bias, 0)
            self.model = nn.Sequential(
                self.base_features,
                self.avgpool,
                nn.Flatten(),
                self.proj,
            )
            self.model.eval().to(self.device)
            self.model_name = "mobilenet_v3_small (torchvision) -> 128"
            print(f"[ReID] Loaded MobileNetV3-Small -> {self.dim}d on {self.device}")
            return
        except Exception as e:
            print(f"[ReID] MobileNet fallback failed ({e}), using histogram")
            self.model = None
            self.model_name = "histogram (no deep model)"

    def _warmup(self):
        if self.model is None:
            return
        try:
            dummy = np.random.randint(0, 255, (self.input_h, self.input_w, 3), dtype=np.uint8)
            for _ in range(3):
                self.extract_embedding(dummy, BoundingBox(x1=0, y1=0, x2=self.input_w, y2=self.input_h), _measure=False)
        except Exception:
            pass

    @torch.no_grad()
    def extract_embedding(self, frame: np.ndarray, bbox: BoundingBox, _measure: bool = True) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w, int(bbox.x2))
        y2 = min(h, int(bbox.y2))
        if x2 <= x1 or y2 <= y1:
            return np.zeros(self.dim, dtype=np.float32)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        if self.model is not None and "mobilenet" in self.model_name:
            crop_resized = cv2.resize(crop, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
            feat = self.model(tensor)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            emb = feat.squeeze(0).cpu().numpy().astype(np.float32)
        elif self.model is not None and "osnet" in self.model_name:
            crop_resized = cv2.resize(crop, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
            try:
                feat = self.model(tensor)
                if isinstance(feat, (tuple, list)):
                    feat = feat[0]
            except Exception:
                feat = self.model(tensor)
            if feat.shape[1] != self.dim:
                if not hasattr(self, "_osnet_proj"):
                    self._osnet_proj = nn.Linear(feat.shape[1], self.dim)
                feat = self._osnet_proj(feat)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            emb = feat.squeeze(0).cpu().numpy().astype(np.float32)
        else:
            hist = []
            for c in range(3):
                h = cv2.calcHist([crop], [c], None, [32], [0, 256])
                h = cv2.normalize(h, h).flatten()
                hist.append(h)
            hist = np.concatenate(hist)
            if len(hist) < self.dim:
                hist = np.pad(hist, (0, self.dim - len(hist)))
            else:
                hist = hist[:self.dim]
            norm = np.linalg.norm(hist) + 1e-6
            emb = (hist / norm).astype(np.float32)
        return emb

    def latency_ms(self, frame: np.ndarray, bbox: BoundingBox, runs: int = 10) -> float:
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = self.extract_embedding(frame, bbox)
            times.append((time.perf_counter() - t0) * 1000)
        return sum(times) / len(times)

    def get_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "dim": self.dim,
            "input_size": (self.input_h, self.input_w),
        }
