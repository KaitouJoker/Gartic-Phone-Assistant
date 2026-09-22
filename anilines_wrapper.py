"""
AniLines 모델 래퍼 모듈
- AniLines (zhenglinpan/AniLines-Anime-Lineart-Extractor) 모델을 
  Gartic Phone Assistant에 통합하기 위한 래퍼
- Basic 모드: RGB 3채널 입력 → 주요 구조 라인아트
- Detail 모드: Grayscale+Sobel 2채널 입력 → 세밀한 라인아트 (배경, 셀 엣지 포함)
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageEnhance

import torch
import torch.nn.functional as F
from torch.amp import autocast

# AniLines 네트워크 모듈 경로 추가
ANILINES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AniLines")
if ANILINES_DIR not in sys.path:
    sys.path.insert(0, ANILINES_DIR)

from network.line_extractor import LineExtractor


def increase_sharpness(img: np.ndarray, factor: float = 6.0) -> np.ndarray:
    """이미지 선명도 증가 (AniLines 원본 코드에서 가져옴)"""
    image = Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(image)
    return np.array(enhancer.enhance(factor))


class AniLinesModel:
    """AniLines 모델을 로드하고 추론하는 클래스"""
    
    def __init__(self, mode: str = "detail", device: str = None, fp16: bool = True):
        """
        Args:
            mode: "basic" 또는 "detail"
            device: "cuda:0", "cpu" 등. None이면 자동 감지
            fp16: FP16 혼합 정밀도 사용 여부 (CUDA에서만 작동)
        """
        self.mode = mode
        self.fp16 = fp16
        
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # CPU에서는 fp16 비활성화
        if "cpu" in self.device:
            self.fp16 = False
        
        self.model = None
    
    def load(self, weights_dir: str = None):
        """모델 가중치를 로드합니다."""
        if weights_dir is None:
            weights_dir = os.path.join(ANILINES_DIR, "weights")
        
        weight_path = os.path.join(weights_dir, f"{self.mode}.pth")
        
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"AniLines {self.mode} 모델 가중치를 찾을 수 없습니다: {weight_path}\n"
                f"가중치 파일을 다운로드하세요:\n"
                f"  Basic: https://drive.google.com/file/d/14Bp8mbQAbiR1rQrEsFp-uNdOou8hoCFr\n"
                f"  Detail: https://drive.google.com/file/d/12U1Mwlonoipk2Yvr12mNaFB30foy420o"
            )
        
        # 모델 생성
        if self.mode == "basic":
            self.model = LineExtractor(3, 1, True).to(self.device)
        elif self.mode == "detail":
            self.model = LineExtractor(2, 1, True).to(self.device)
        else:
            raise ValueError(f"지원하지 않는 모드: {self.mode}. 'basic' 또는 'detail'을 사용하세요.")
        
        # 가중치 로드
        self.model.load_state_dict(
            torch.load(weight_path, map_location=torch.device(self.device), weights_only=True)
        )
        
        # 추론 모드
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        return self
    
    def inference(self, pil_image: Image.Image, binarize: float = -1) -> np.ndarray:
        """
        PIL 이미지를 입력받아 라인아트를 추출합니다.
        
        Args:
            pil_image: 입력 PIL Image (RGB)
            binarize: 이진화 임계값 (0~1). -1이면 이진화 비활성화 (그레이스케일 출력)
            
        Returns:
            np.ndarray: 그레이스케일 라인아트 이미지 (0-255, uint8)
                       흰 배경에 검은 선 (255=배경, 0=선) — 반전하여 반환
        """
        if self.model is None:
            self.load()
        
        # PIL Image → BGR numpy (AniLines 원본 코드 호환)
        img = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # 모드별 전처리
        if self.mode == "basic":
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_sharp = increase_sharpness(img_rgb)
            x_in = torch.from_numpy(img_sharp).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        else:  # detail
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = cv2.magnitude(sobelx, sobely)
            sobel = 255 - cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            
            img_tensor = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0).float().to(self.device) / 255.0
            sobel_tensor = torch.from_numpy(sobel).unsqueeze(0).unsqueeze(0).float().to(self.device) / 255.0
            
            x_in = torch.cat([img_tensor, sobel_tensor], dim=1)
        
        # 8의 배수로 패딩
        B, C, H, W = x_in.shape
        pad_h = (8 - (H % 8)) % 8
        pad_w = (8 - (W % 8)) % 8
        if pad_h > 0 or pad_w > 0:
            x_in = F.pad(x_in, (0, pad_w, 0, pad_h), mode='reflect')
        
        # 추론
        with torch.no_grad(), autocast(enabled=self.fp16, device_type='cuda' if 'cuda' in self.device else 'cpu'):
            pred = self.model(x_in)
        
        # 패딩 제거
        pred = pred[:, :, :H, :W]
        
        # 이진화 (선택적)
        if binarize != -1 and 0 <= binarize <= 1:
            pred = (pred > binarize).float()
        
        # 출력: 흰 배경에 검은 선 (기존 파이프라인 호환)
        # AniLines 원본 출력은 흰색=선, 검은색=배경이므로 반전
        result = np.clip((pred[0, 0].cpu().numpy() * 255) + 0.5, 0, 255).astype(np.uint8)
        
        # 반전: 흰 배경(255)에 검은 선(0) → 기존 contour 추출 파이프라인과 호환
        # AniLines 출력에서 높은 값 = 선이므로, 컨투어를 찾으려면 반전 필요
        # 하지만 기존 코드에서 threshold로 이진화 후 findContours 하므로 그대로 반환
        return result

    def to(self, device: str):
        """모델을 지정된 디바이스로 이동"""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self


# 전역 캐시
_anilines_cache = {}

def get_anilines_model(mode: str = "detail", device: str = None, logger=None) -> AniLinesModel:
    """
    AniLines 모델을 캐시에서 가져오거나 새로 로드합니다.
    기존 model_cache 시스템과 별도로 관리합니다.
    """
    cache_key = f"anilines_{mode}"
    
    if cache_key in _anilines_cache:
        if logger:
            logger.info(f"캐시에서 AniLines ({mode}) 모델을 로드했습니다.")
        return _anilines_cache[cache_key]
    
    if logger:
        logger.info(f"AniLines ({mode}) 모델 로딩 중... (첫 실행 시 시간이 걸릴 수 있습니다)")
    
    model = AniLinesModel(mode=mode, device=device)
    model.load()
    
    _anilines_cache[cache_key] = model
    
    if logger:
        logger.info(f"AniLines ({mode}) 모델 로딩 완료. 디바이스: {model.device}")
    
    return model
