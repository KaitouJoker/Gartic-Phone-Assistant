"""
Anime2Sketch 모델 래퍼 모듈
- Mukosame/Anime2Sketch (U-Net 기반 스케치 추출) 모델을 Gartic Phone Assistant에 통합
- 입력: PIL Image (RGB)
- 출력: 1픽셀 스켈레톤화된 라인아트 (255=선, 0=배경)
"""

import os
import sys
import urllib.request
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from skimage.morphology import skeletonize

A2S_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anime2sketch_repo")
if A2S_DIR not in sys.path:
    sys.path.insert(0, A2S_DIR)

try:
    from model import UnetGenerator, functools
    A2S_AVAILABLE = True
except ImportError:
    A2S_AVAILABLE = False


class Anime2SketchModel:
    """Anime2Sketch 모델 로더 및 추론 클래스"""

    def __init__(self, device: str = None, load_size: int = 512):
        self.load_size = load_size
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = None

    def load(self, weights_path: str = None):
        """가중치 파일을 로드합니다 (부재 시 자동 다운로드 지원)."""
        if weights_path is None:
            weights_dir = os.path.join(A2S_DIR, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            weights_path = os.path.join(weights_dir, "netG.pth")

        if not os.path.exists(weights_path):
            url = "https://huggingface.co/gyrojeff/Anime2Sketch/resolve/main/netG.pth"
            print(f"[Anime2Sketch] 가중치 다운로드 중: {url} -> {weights_path}")
            urllib.request.urlretrieve(url, weights_path)

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        net = UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer, use_dropout=False)
        ckpt = torch.load(weights_path, map_location="cpu")
        for key in list(ckpt.keys()):
            if "module." in key:
                ckpt[key.replace("module.", "")] = ckpt[key]
                del ckpt[key]
        net.load_state_dict(ckpt)
        self.model = net.to(self.device).eval()

    def inference(self, pil_image: Image.Image, threshold: float = 0.4, return_skeleton: bool = False) -> np.ndarray:
        """
        이미지에서 스케치 선화를 추출하여 반환합니다.
        
        Args:
            pil_image: PIL Image (RGB)
            threshold: 선 검출 임계값 (0.0 ~ 1.0, 기본값 0.4)
            return_skeleton: True이면 1픽셀 스켈레톤화 적용, False이면 두께/면이 보존된 바이너리 마스크 반환
        
        Returns:
            np.ndarray: uint8 1채널 엣지 맵 (255=선/면, 0=배경)
        """
        if self.model is None:
            self.load()

        orig_w, orig_h = pil_image.size
        # 32의 배수로 맞추거나 load_size로 리사이즈
        img_resized = pil_image.convert("RGB").resize((self.load_size, self.load_size), Image.BICUBIC)
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        img_t = (img_t - 0.5) / 0.5  # [-1, 1] 범위로 정규화

        with torch.no_grad():
            out = self.model(img_t.to(self.device))
            out_np = out[0, 0].cpu().numpy()
            # [-1, 1] -> [0, 255]
            out_np = (out_np + 1.0) / 2.0 * 255.0
            out_np = np.clip(out_np, 0, 255).astype(np.uint8)

        # 원래 크기로 복원
        sketch = cv2.resize(out_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        # 모델 출력은 흰색 배경(~255), 어두운 선(~0)이므로 반전 (선=높은 값, 배경=0)
        inverted = 255 - sketch

        # 임계값 적용: threshold 파라미터 활용 (기본 0.4 -> 약 102 이상인 선 검출)
        t_val = int(255 * threshold)
        _, binary_edges = cv2.threshold(inverted, t_val, 255, cv2.THRESH_BINARY)

        if return_skeleton:
            skeleton = skeletonize(binary_edges > 0)
            return (skeleton * 255).astype(np.uint8)

        return binary_edges


_anime2sketch_cache = {}

def get_anime2sketch_model(device: str = None, logger=None) -> Anime2SketchModel:
    """Anime2Sketch 싱글톤 인스턴스 반환"""
    cache_key = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if cache_key in _anime2sketch_cache:
        if logger:
            logger.info("캐시에서 Anime2Sketch 모델을 로드했습니다.")
        return _anime2sketch_cache[cache_key]

    if logger:
        logger.info("Anime2Sketch 모델 로딩 중...")
    model = Anime2SketchModel(device=device)
    model.load()
    if logger:
        logger.info(f"Anime2Sketch 모델 로딩 완료. 디바이스: {model.device}")
    _anime2sketch_cache[cache_key] = model
    return model
