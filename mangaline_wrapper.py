"""
MangaLineExtraction 모델 래퍼 모듈
- Chengze Li et al. 'Deep Extraction of Manga Structural Lines'
- PyTorch res_skip 아키텍처 및 erika.pth 가중치 활용
- 입력: PIL Image (RGB 또는 Grayscale)
- 출력: 1픽셀 스켈레톤화된 구조 선화 (255=선, 0=배경)
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from skimage.morphology import skeletonize
from huggingface_hub import hf_hub_download


class _bn_relu_conv(nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super(_bn_relu_conv, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_filters, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample, padding=(fw // 2, fh // 2), padding_mode='zeros')
        )

    def forward(self, x):
        return self.model(x)


class _u_bn_relu_conv(nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super(_u_bn_relu_conv, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_filters, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample, padding=(fw // 2, fh // 2)),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        return self.model(x)


class _shortcut(nn.Module):
    def __init__(self, in_filters, nb_filters, subsample=1):
        super(_shortcut, self).__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters or subsample != 1:
            self.process = True
            self.model = nn.Sequential(
                nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample)
            )

    def forward(self, x, y):
        if self.process:
            return self.model(x) + y
        else:
            return x + y


class _u_shortcut(nn.Module):
    def __init__(self, in_filters, nb_filters, subsample):
        super(_u_shortcut, self).__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters:
            self.process = True
            self.model = nn.Sequential(
                nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample, padding_mode='zeros'),
                nn.Upsample(scale_factor=2, mode='nearest')
            )

    def forward(self, x, y):
        if self.process:
            return self.model(x) + y
        else:
            return x + y


class basic_block(nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super(basic_block, self).__init__()
        self.conv1 = _bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        return self.shortcut(x, self.residual(self.conv1(x)))


class _u_basic_block(nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super(_u_basic_block, self).__init__()
        self.conv1 = _u_bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _u_shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        return self.shortcut(x, self.residual(self.conv1(x)))


class _residual_block(nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions, is_first_layer=False):
        super(_residual_block, self).__init__()
        layers = []
        for i in range(repetitions):
            init_subsample = 1
            if i == repetitions - 1 and not is_first_layer:
                init_subsample = 2
            layers.append(basic_block(in_filters=in_filters if i == 0 else nb_filters, nb_filters=nb_filters, init_subsample=init_subsample))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class _upsampling_residual_block(nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions):
        super(_upsampling_residual_block, self).__init__()
        layers = []
        for i in range(repetitions):
            if i == 0:
                layers.append(_u_basic_block(in_filters=in_filters, nb_filters=nb_filters))
            else:
                layers.append(basic_block(in_filters=nb_filters, nb_filters=nb_filters))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class res_skip(nn.Module):
    """MangaLineExtraction Skip-Residual 신경망"""

    def __init__(self):
        super(res_skip, self).__init__()
        self.block0 = _residual_block(in_filters=1, nb_filters=24, repetitions=2, is_first_layer=True)
        self.block1 = _residual_block(in_filters=24, nb_filters=48, repetitions=3)
        self.block2 = _residual_block(in_filters=48, nb_filters=96, repetitions=5)
        self.block3 = _residual_block(in_filters=96, nb_filters=192, repetitions=7)
        self.block4 = _residual_block(in_filters=192, nb_filters=384, repetitions=12)

        self.block5 = _upsampling_residual_block(in_filters=384, nb_filters=192, repetitions=7)
        self.res1 = _shortcut(in_filters=192, nb_filters=192)

        self.block6 = _upsampling_residual_block(in_filters=192, nb_filters=96, repetitions=5)
        self.res2 = _shortcut(in_filters=96, nb_filters=96)

        self.block7 = _upsampling_residual_block(in_filters=96, nb_filters=48, repetitions=3)
        self.res3 = _shortcut(in_filters=48, nb_filters=48)

        self.block8 = _upsampling_residual_block(in_filters=48, nb_filters=24, repetitions=2)
        self.res4 = _shortcut(in_filters=24, nb_filters=24)

        self.block9 = _residual_block(in_filters=24, nb_filters=16, repetitions=2, is_first_layer=True)
        self.conv15 = _bn_relu_conv(in_filters=16, nb_filters=1, fh=1, fw=1, subsample=1)

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x5 = self.block5(x4)
        res1 = self.res1(x3, x5)

        x6 = self.block6(res1)
        res2 = self.res2(x2, x6)

        x7 = self.block7(res2)
        res3 = self.res3(x1, x7)

        x8 = self.block8(res3)
        res4 = self.res4(x0, x8)

        x9 = self.block9(res4)
        return self.conv15(x9)


class MangaLineModel:
    """MangaLineExtraction 모델 로더 및 추론 클래스"""

    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = None

    def load(self, weights_path: str = None):
        """가중치 파일을 로드합니다 (부재 시 HuggingFace에서 자동 다운로드)."""
        if weights_path is None or not os.path.exists(weights_path):
            weights_path = hf_hub_download(repo_id="lllyasviel/Annotators", filename="erika.pth")

        net = res_skip()
        ckpt = torch.load(weights_path, map_location="cpu")
        net.load_state_dict(ckpt)
        self.model = net.to(self.device).eval()

    def inference(self, pil_image: Image.Image, threshold: float = 0.4, return_skeleton: bool = False) -> np.ndarray:
        """
        만화 구조 라인을 추출하여 반환합니다.
        
        Args:
            pil_image: PIL Image
            threshold: 선 검출 임계값 (0.0 ~ 1.0, 기본값 0.4)
            return_skeleton: True이면 1픽셀 스켈레톤화 적용, False이면 두께/면이 보존된 바이너리 마스크 반환
            
        Returns:
            np.ndarray: uint8 1채널 엣지 맵 (255=선/면, 0=배경)
        """
        if self.model is None:
            self.load()

        # 그레이스케일 변환
        img_gray = np.array(pil_image.convert("L"), dtype=np.float32)
        h, w = img_gray.shape

        # 16배수 패딩
        rows = int(np.ceil(h / 16.0)) * 16
        cols = int(np.ceil(w / 16.0)) * 16
        patch = np.ones((1, 1, rows, cols), dtype=np.float32) * 255.0
        patch[0, 0, 0:h, 0:w] = img_gray

        with torch.no_grad():
            tensor = torch.from_numpy(patch).to(self.device)
            y = self.model(tensor)
            yc = y.cpu().numpy()[0, 0, 0:h, 0:w]
            yc = np.clip(yc, 0, 255).astype(np.uint8)

        # 모델 출력: 흰색 배경(~255), 검은 펜선(~0) -> 반전
        inverted = 255 - yc

        # 임계값 적용 (기본 0.4 -> 102 이상인 선 검출)
        t_val = int(255 * threshold)
        _, binary_edges = cv2.threshold(inverted, t_val, 255, cv2.THRESH_BINARY)

        if return_skeleton:
            skeleton = skeletonize(binary_edges > 0)
            return (skeleton * 255).astype(np.uint8)

        return binary_edges


_mangaline_cache = {}

def get_mangaline_model(device: str = None, logger=None) -> MangaLineModel:
    """MangaLineExtraction 싱글톤 인스턴스 반환"""
    cache_key = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if cache_key in _mangaline_cache:
        if logger:
            logger.info("캐시에서 MangaLineExtraction 모델을 로드했습니다.")
        return _mangaline_cache[cache_key]

    if logger:
        logger.info("MangaLineExtraction 모델 로딩 중...")
    model = MangaLineModel(device=device)
    model.load()
    if logger:
        logger.info(f"MangaLineExtraction 모델 로딩 완료. 디바이스: {model.device}")
    _mangaline_cache[cache_key] = model
    return model
