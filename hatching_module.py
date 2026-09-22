"""
hatching_module.py
- 면(채색 영역 및 굵은 선)을 만화/일러스트 스타일의 빗금(Hatching) 및 Cross-Contour(등고선) 기법으로 채우는 모듈
- 적응형 명암 빗금(Adaptive Hatching): 원본 이미지의 명암에 따라 빗금 간격과 밀도를 자동으로 조절
- 지원 모드:
  1. 외곽선 + 빗금 (Outline + Hatching) - 테두리 1px + 내부 빗금
  2. 스켈레톤화 (Skeletonize) - 기존 1픽셀 중심선 추출
  3. 순수 빗금 (Hatching Only) - 테두리 없이 빗금으로만 표현
- 지원 패턴:
  1. 45° (대각선)
  2. 135° (역대각선)
  3. 0° (수평)
  4. 90° (수직)
  5. 크로스 빗금 (Cross) - 45° + 135° 대각선 X자 교차
  6. 격자 빗금 (Grid) - 0° + 90° 수평/수직 직교 바둑판 모눈 격자
  7. Cross-Contour (등고선)
"""

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize


def generate_parallel_hatch_mask(mask: np.ndarray, spacing: int = 8, angle_type: str = "45° (대각선)") -> np.ndarray:
    """
    지정된 각도와 간격으로 마스크 내부에 평행 빗금 선분을 생성합니다.
    - 45°: 우상향 대각선
    - 135°: 좌상향 역대각선
    - 0°/수평: 가로선
    - 90°/수직: 세로선
    - 격자/Grid: 0° 수평 + 90° 수직 직교 바둑판 모눈 격자 (+)
    - 크로스/Cross: 45° 대각선 + 135° 역대각선 X자 교차선 (X)
    """
    h, w = mask.shape
    y, x = np.indices((h, w))
    spacing = max(2, int(spacing))
    grid = np.zeros((h, w), dtype=bool)

    if "45°" in angle_type:
        grid = ((x + y) % spacing == 0)
    elif "135°" in angle_type:
        grid = ((x - y) % spacing == 0)
    elif "0°" in angle_type or "수평" in angle_type:
        grid = (y % spacing == 0)
    elif "90°" in angle_type or "수직" in angle_type:
        grid = (x % spacing == 0)
    elif "격자" in angle_type or "Grid" in angle_type:
        # 바둑판 모눈 격자 (0° 수평 + 90° 수직 직교)
        grid = (y % spacing == 0) | (x % spacing == 0)
    elif "크로스" in angle_type or "Cross" in angle_type:
        # 정통 크로스해칭 (45° 대각선 + 135° 역대각선 X자 교차)
        grid = ((x + y) % spacing == 0) | ((x - y) % spacing == 0)
    else:
        grid = ((x + y) % spacing == 0)

    hatch = grid & (mask > 0)
    return (hatch.astype(np.uint8) * 255)


def generate_cross_contour_mask(mask: np.ndarray, spacing: int = 8) -> np.ndarray:
    """
    형태의 3D 곡면을 감싸는 거리장(Distance Transform) 기반 등고선 루프 곡선을 생성합니다.
    """
    h, w = mask.shape
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_d = np.max(dist)
    canvas = np.zeros((h, w), dtype=np.uint8)

    if max_d < 1.5:
        return canvas

    # 선 두께(반경)에 비례하여 동심 등고선 간격 산출
    step = max(1.5, float(spacing) * 0.25)
    d = step
    while d < max_d:
        ring = (dist >= d).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(ring, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, 255, 1)
        d += step

    return canvas


def apply_adaptive_hatching_mask(mask: np.ndarray, orig_gray: np.ndarray, base_spacing: int = 8, pattern: str = "45° (대각선)") -> np.ndarray:
    """
    원본 이미지의 명암(Darkness)에 연동하여 차등적인 빗금 간격과 밀도를 생성합니다.
    - 밝은 음영: 넓은 간격 (base_spacing * 2)
    - 중간 음영: 기본 간격 (base_spacing)
    - 짙은 음영: 좁은 간격 (base_spacing // 2) + 교차선
    """
    h, w = mask.shape
    if orig_gray.shape != (h, w):
        orig_gray = cv2.resize(orig_gray, (w, h), interpolation=cv2.INTER_AREA)

    darkness = 255 - orig_gray
    base_spacing = max(2, int(base_spacing))

    if "Cross-Contour" in pattern or "등고선" in pattern:
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_d = np.max(dist)
        canvas = np.zeros((h, w), dtype=np.uint8)
        if max_d >= 1.5:
            step = max(1.5, float(base_spacing) * 0.25)
            d = step
            while d < max_d:
                ring = (dist >= d).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(ring, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                temp = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(temp, cnts, -1, 255, 1)
                # 밝은 하이라이트(darkness < 50) 영역은 깔끔하게 비우고, 음영 영역(darkness >= 50)에 등고선 곡선 배치
                canvas[(temp == 255) & (darkness >= 50)] = 255
                d += step
        # 짙은 영역(darkness >= 170)에는 보조 대각 빗금 추가하여 깊은 명암 보강
        deep_dark_mask = (darkness >= 170) & (mask > 0)
        y, x = np.indices((h, w))
        diag_fill = ((x + y) % base_spacing == 0) & deep_dark_mask
        canvas[diag_fill] = 255
        return canvas

    y, x = np.indices((h, w))
    diag1 = (x + y)
    diag2 = (x - y)

    s1 = base_spacing * 2
    s2 = base_spacing
    s3 = max(2, base_spacing // 2)

    # 1. 격자 빗금 (수평 0° + 수직 90° 직교 바둑판 패턴)
    if "격자" in pattern or "Grid" in pattern:
        m1 = (darkness >= 60) & (mask > 0) & ((y % s1 == 0) | (x % s1 == 0))
        m2 = (darkness >= 120) & (mask > 0) & ((y % s2 == 0) | (x % s2 == 0))
        m3 = (darkness >= 175) & (mask > 0) & ((y % s3 == 0) | (x % s3 == 0))
        m4 = (darkness >= 230) & (mask > 0) & ((y % s3 == 0) | (x % s3 == 0))
        combined_bool = m1 | m2 | m3 | m4
        return (combined_bool.astype(np.uint8) * 255)

    # 2. 크로스 빗금 (45° 대각선 + 135° 역대각선 X자 교차 패턴)
    if "크로스" in pattern or "Cross" in pattern:
        m1 = (darkness >= 60) & (mask > 0) & ((diag1 % s1 == 0) | (diag2 % s1 == 0))
        m2 = (darkness >= 120) & (mask > 0) & ((diag1 % s2 == 0) | (diag2 % s2 == 0))
        m3 = (darkness >= 175) & (mask > 0) & ((diag1 % s3 == 0) | (diag2 % s3 == 0))
        m4 = (darkness >= 230) & (mask > 0) & ((diag1 % s3 == 0) | (diag2 % s3 == 0))
        combined_bool = m1 | m2 | m3 | m4
        return (combined_bool.astype(np.uint8) * 255)

    # 3. 단일 방향 평행 빗금 (0°, 90°, 135°, 45°)
    if "0°" in pattern or "수평" in pattern:
        axis1, axis2 = y, x
    elif "90°" in pattern or "수직" in pattern:
        axis1, axis2 = x, y
    elif "135°" in pattern:
        axis1, axis2 = diag2, diag1
    else:
        axis1, axis2 = diag1, diag2

    # Level 1: 밝은 음영 (darkness >= 60) -> 단일 주축 넓은 간격
    m1 = (darkness >= 60) & (mask > 0) & (axis1 % s1 == 0)
    # Level 2: 중간 음영 (darkness >= 120) -> 단일 주축 기본 간격
    m2 = (darkness >= 120) & (mask > 0) & (axis1 % s2 == 0)
    # Level 3: 깊은 그림자 (darkness >= 175) -> 교차 축 빗금 보조 추가
    m3 = (darkness >= 175) & (mask > 0) & (axis2 % s2 == 0)
    # Level 4: 극암부/먹칠 (darkness >= 230) -> 촘촘한 간격
    m4 = (darkness >= 230) & (mask > 0) & (axis1 % s3 == 0)

    combined_bool = m1 | m2 | m3 | m4
    return (combined_bool.astype(np.uint8) * 255)


def process_surface_with_hatching(
    binary_mask: np.ndarray,
    orig_image: Image.Image = None,
    mode: str = "외곽선 + 빗금",
    pattern: str = "45° (대각선)",
    spacing: int = 8,
    adaptive: bool = True
) -> np.ndarray:
    """
    이진 마스크를 사용자가 지정한 빗금 모드 및 패턴에 따라 처리하여 최종 1픽셀 라인아트를 반환합니다.
    
    Args:
        binary_mask: uint8 1채널 마스크 (255=선/면, 0=배경)
        orig_image: PIL Image 원본 (적응형 명암 분석에 사용)
        mode: "외곽선 + 빗금", "스켈레톤화", "순수 빗금"
        pattern: "45° (대각선)", "135° (역대각선)", "0° (수평)", "90° (수직)", "Cross (격자 빗금)", "Cross-Contour (등고선)"
        spacing: 빗금 간격 (px)
        adaptive: True이면 원본 이미지 명암에 따라 간격 자동 조절
        
    Returns:
        np.ndarray: uint8 1채널 엣지 맵 (255=그릴 선, 0=배경)
    """
    if binary_mask is None or np.sum(binary_mask > 0) == 0:
        return binary_mask

    # 1. 스켈레톤화 모드인 경우: 기존 방식대로 1픽셀 중심선만 추출
    if mode == "스켈레톤화":
        skeleton = skeletonize(binary_mask > 0)
        return (skeleton * 255).astype(np.uint8)

    h, w = binary_mask.shape

    # 2. 외곽선(Outline) 추출
    outline_canvas = np.zeros((h, w), dtype=np.uint8)
    if mode in ("외곽선 + 빗금", "해칭 없음", "외곽선만", "외곽선만 (해칭 없음)", "해칭 없음 (외곽선만)"):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(outline_canvas, contours, -1, 255, 1)

    # 3. 해칭 없음인 경우: 외곽선만 즉시 반환
    if mode in ("해칭 없음", "외곽선만", "외곽선만 (해칭 없음)", "해칭 없음 (외곽선만)") or pattern in ("해칭 없음", "없음"):
        return outline_canvas

    # 4. 빗금(Hatching) 생성
    hatch_canvas = np.zeros((h, w), dtype=np.uint8)
    orig_gray = None
    if orig_image is not None:
        orig_gray = np.array(orig_image.convert("L"))

    if adaptive and orig_gray is not None:
        hatch_canvas = apply_adaptive_hatching_mask(binary_mask, orig_gray, base_spacing=spacing, pattern=pattern)
    else:
        if "Cross-Contour" in pattern or "등고선" in pattern:
            hatch_canvas = generate_cross_contour_mask(binary_mask, spacing=spacing)
        else:
            hatch_canvas = generate_parallel_hatch_mask(binary_mask, spacing=spacing, angle_type=pattern)

    # 5. 모드에 따라 결합
    if mode == "외곽선 + 빗금":
        result = cv2.bitwise_or(outline_canvas, hatch_canvas)
    elif mode == "순수 빗금":
        result = hatch_canvas
    else:
        result = outline_canvas

    return result
