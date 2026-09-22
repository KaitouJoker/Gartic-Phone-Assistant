import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import Toplevel, Canvas
from PIL import Image, ImageTk, ImageGrab, ImageDraw, ImageOps
import pyautogui
import time
import numpy as np
import cv2
import math
import logging
import sys

import torch
from controlnet_aux import LineartAnimeDetector, AnylineDetector
from skimage.morphology import skeletonize

# 라인아트 모델 래퍼
try:
    from anilines_wrapper import get_anilines_model
    ANILINES_AVAILABLE = True
except ImportError:
    ANILINES_AVAILABLE = False

try:
    from anime2sketch_wrapper import get_anime2sketch_model
    A2S_AVAILABLE = True
except ImportError:
    A2S_AVAILABLE = False

try:
    from mangaline_wrapper import get_mangaline_model
    MANGALINE_AVAILABLE = True
except ImportError:
    MANGALINE_AVAILABLE = False

from hatching_module import process_surface_with_hatching

# --- Platform-Specific Mouse Controller ---

class MouseController:
    """Base class for mouse controllers."""
    def move_to(self, x, y, duration=0):
        raise NotImplementedError
    def mouse_down(self, x, y):
        raise NotImplementedError
    def mouse_up(self, x, y):
        raise NotImplementedError

class PyAutoGUIMouseController(MouseController):
    """Mouse control using pyautogui."""
    def move_to(self, x, y, duration=0):
        pyautogui.moveTo(x, y, duration=duration, _pause=False)
    def mouse_down(self, x, y):
        pyautogui.mouseDown(_pause=False)
    def mouse_up(self, x, y):
        pyautogui.mouseUp(_pause=False)

class Win32MouseController(MouseController):
    """Mouse control using win32api for higher performance."""
    def __init__(self):
        import win32api, win32con
        self.win32api = win32api
        self.win32con = win32con

    def move_to(self, x, y, duration=0): # duration is ignored, but kept for API compatibility
        self.win32api.SetCursorPos((x, y))
    def mouse_down(self, x, y):
        self.win32api.mouse_event(self.win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    def mouse_up(self, x, y):
        self.win32api.mouse_event(self.win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

def get_mouse_controller(logger):
    """Factory function to get the best available mouse controller."""
    if sys.platform == 'win32':
        try:
            controller = Win32MouseController()
            logger.info("Windows 환경 감지: 고성능 win32api 마우스 컨트롤러를 사용합니다.")
            return controller
        except ImportError:
            logger.warning("pywin32 라이브러리를 찾을 수 없습니다. pyautogui로 대체합니다. (pip install pywin32)")
            return PyAutoGUIMouseController()
    else:
        logger.info(f"{sys.platform} 환경 감지: pyautogui 마우스 컨트롤러를 사용합니다.")
        return PyAutoGUIMouseController()

# --- End of Mouse Controller Section ---


model_cache = {}
CONTOUR_MODES = {"외곽선만 찾기": cv2.RETR_EXTERNAL, "모든 선 찾기": cv2.RETR_LIST, "모든 선 찾기 + 계층": cv2.RETR_TREE}
CONTOUR_METHODS = {"선 압축하기": cv2.CHAIN_APPROX_SIMPLE, "모든 점 저장하기": cv2.CHAIN_APPROX_NONE}

def load_processor_model(model_name, logger):
    if model_name in model_cache: 
        logger.info(f"캐시에서 {model_name} 모델을 로드했습니다.")
        return model_cache[model_name]
    
    logger.info(f"{model_name} 모델 로딩 중... (첫 실행 시 시간이 걸릴 수 있습니다)")
    model = None
    try:
        if model_name == "Lineart Anime": 
            model = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
        elif model_name == "AnyLine":
            model = AnylineDetector.from_pretrained("TheMistoAI/MistoLine", filename="MTEED.pth", subfolder="Anyline")
        elif model_name == "Anime2Sketch":
            if not A2S_AVAILABLE:
                logger.error("Anime2Sketch 모듈을 찾을 수 없습니다. anime2sketch_wrapper.py를 확인하세요.")
                return None
            model = get_anime2sketch_model(logger=logger)
            model_cache[model_name] = model
            return model
        elif model_name == "MangaLineExtraction":
            if not MANGALINE_AVAILABLE:
                logger.error("MangaLineExtraction 모듈을 찾을 수 없습니다. mangaline_wrapper.py를 확인하세요.")
                return None
            model = get_mangaline_model(logger=logger)
            model_cache[model_name] = model
            return model
        elif model_name in ("AniLines Basic", "AniLines Detail"):
            if not ANILINES_AVAILABLE:
                logger.error("AniLines 모듈을 찾을 수 없습니다. anilines_wrapper.py를 확인하세요.")
                return None
            mode = "basic" if model_name == "AniLines Basic" else "detail"
            model = get_anilines_model(mode=mode, logger=logger)
            model_cache[model_name] = model
            return model
        
        if model: 
            logger.info(f"{model_name} 모델 로딩 완료.")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model_cache[model_name] = model
        return model
    except Exception as e: 
        logger.exception(f"{model_name} 모델 로딩 실패: {e}")
        return None

def load_settings():
    settings = {};
    try:
        with open("line_draw_setting.txt", "r", encoding="utf-8") as f:
            for line in f: key, value = line.strip().split("=", 1); settings[key] = value
    except FileNotFoundError: return {}
    return settings

def save_settings(app):
    settings = {
        "CANVAS_AREA": app.canvas_area_var.get(),
        "IMAGE_PATH": app.image_path_var.get(),
        "PRECISION": app.precision_var.get(),
        "LINE_EPSILON": app.line_epsilon_var.get(),
        "LINE_DELAY": app.line_delay_var.get(),
        "MOUSE_DURATION": app.mouse_duration_var.get(),
        "MOUSE_MOVES_PER_SECOND": app.mouse_moves_per_second_var.get(),
        "DENOISE_FILTER": app.denoise_filter_var.get(),
        "CONTOUR_MODE": app.contour_mode_var.get(),
        "CONTOUR_METHOD": app.contour_method_var.get(),
        "COMBINATION_METHOD": app.combination_method_var.get(),
        "NUM_LAYERS": app.num_layers_var.get(),
        "HATCH_MODE": getattr(app, "hatch_mode_var", None).get() if hasattr(app, "hatch_mode_var") else "외곽선 + 빗금",
        "HATCH_PATTERN": getattr(app, "hatch_pattern_var", None).get() if hasattr(app, "hatch_pattern_var") else "45° (대각선)",
        "HATCH_SPACING": getattr(app, "hatch_spacing_var", None).get() if hasattr(app, "hatch_spacing_var") else "8",
        "HATCH_ADAPTIVE": getattr(app, "hatch_adaptive_var", None).get() if hasattr(app, "hatch_adaptive_var") else True,
    }
    for i, layer in enumerate(app.layers):
        settings[f"L{i+1}_ENABLED"] = layer["enabled"].get()
        settings[f"L{i+1}_MODEL"] = layer["model"].get()
        settings[f"L{i+1}_THRESHOLD"] = layer["threshold"].get()
    with open("line_draw_setting.txt", "w", encoding="utf-8") as f:
        for key, value in settings.items(): f.write(f"{key}={value}\n")

class AreaSelector:
    def __init__(self, root):
        self.root, self.area, self.start_x, self.start_y, self.rect = root, None, None, None, None
        self.top = Toplevel(root); self.top.attributes("-fullscreen", True); self.top.attributes("-alpha", 0.3); self.top.overrideredirect(True)
        self.canvas = Canvas(self.top, cursor="cross", bg="grey"); self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press); self.canvas.bind("<B1-Motion>", self.on_mouse_drag); self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
    def on_button_press(self, event): self.start_x, self.start_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
    def on_mouse_drag(self, event):
        if not self.rect: self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)
        self.canvas.coords(self.rect, self.start_x, self.start_y, self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
    def on_button_release(self, event):
        x1, y1, x2, y2 = min(self.start_x, event.x), min(self.start_y, event.y), max(self.start_x, event.x), max(self.start_y, event.y)
        self.area = (int(x1), int(y1), int(x2), int(y2)); self.top.destroy()
def select_area(root):
    selector = AreaSelector(root); root.wait_window(selector.top)
    return (selector.area, ImageGrab.grab(bbox=selector.area)) if selector.area else (None, None)

def crop_image_to_content(image_obj, logger):
    inverted_image = ImageOps.invert(image_obj.convert('L')); bbox = inverted_image.getbbox()
    if bbox:
        padding = 20; left, upper, right, lower = bbox; left, upper = max(0, left - padding), max(0, upper - padding)
        right, lower = min(image_obj.width, right + padding), min(image_obj.height, lower + padding)
        logger.info(f"미리보기 이미지를 내용에 맞게 자릅니다."); return image_obj.crop((left, upper, right, lower))
    return image_obj

# 라인아트 모델인지 확인하는 헬퍼
def is_lineart_model(model_name):
    return model_name in (
        "AniLines Detail",
        "AniLines Basic",
        "Lineart Anime",
        "Anime2Sketch",
        "AnyLine",
        "MangaLineExtraction"
    )

def get_edges_from_model(model_name, image, threshold, logger):
    edges = None
    target_size = image.size

    if model_name in ("AniLines Basic", "AniLines Detail"):
        model = load_processor_model(model_name, logger)
        if model:
            logger.info(f"{model_name} 모델 추론 (Binarize: {threshold})...")
            binarize_val = threshold if threshold < 1.0 else -1
            result = model.inference(image, binarize=binarize_val)
            if binarize_val == -1:
                inverted = 255 - result
                _, edges = cv2.threshold(inverted, 30, 255, cv2.THRESH_BINARY)
            else:
                edges = 255 - result
            
            _, binary_edges = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
            edges = binary_edges

    elif model_name == "Anime2Sketch":
        model = load_processor_model(model_name, logger)
        if model:
            logger.info(f"Anime2Sketch 모델 추론 (Threshold: {threshold})...")
            edges = model.inference(image, threshold=threshold, return_skeleton=False)

    elif model_name == "MangaLineExtraction":
        model = load_processor_model(model_name, logger)
        if model:
            logger.info(f"MangaLineExtraction 모델 추론 (Threshold: {threshold})...")
            edges = model.inference(image, threshold=threshold, return_skeleton=False)

    elif model_name == "AnyLine":
        model = load_processor_model(model_name, logger)
        if model:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            logger.info(f"AnyLine 모델 추론 (Threshold: {threshold})...")
            with torch.no_grad():
                res_img = model(image)
            edge_map = np.array(res_img.convert("L"))
            t_val = int(255 * threshold)
            _, binary_edges = cv2.threshold(edge_map, t_val, 255, cv2.THRESH_BINARY)
            edges = binary_edges

    elif model_name == "Lineart Anime":
        model = load_processor_model(model_name, logger)
        if model:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            logger.info(f"Lineart Anime 모델 추론 (Threshold: {threshold})...")
            with torch.no_grad():
                res_img = model(image, safe_steps=True)
            edge_map = np.array(res_img)
            if len(edge_map.shape) == 3:
                edge_map = cv2.cvtColor(edge_map, cv2.COLOR_RGB2GRAY)
            t_val = int(255 * threshold)
            _, binary_edges = cv2.threshold(edge_map, t_val, 255, cv2.THRESH_BINARY)
            edges = binary_edges

    else:
        model = load_processor_model(model_name, logger)
        if model:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            logger.info(f"{model_name} 모델 추론 (Threshold: {threshold})...")
            with torch.no_grad():
                edge_map = np.array(model(image))
            if len(edge_map.shape) == 3:
                edge_map = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)
            edges = (edge_map > int(255 * threshold)).astype(np.uint8) * 255

    if edges is not None:
        if len(edges.shape) == 3:
            edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
        if edges.shape[1] != target_size[0] or edges.shape[0] != target_size[1]:
            logger.warning(f"{model_name} 모델 출력 크기({edges.shape[1]}x{edges.shape[0]})가 분석 크기와 다릅니다. 강제 조정합니다.")
            edges = cv2.resize(edges, target_size, interpolation=cv2.INTER_AREA)

    return edges

def calculate_segment_length(segment):
    length = 0.0
    for i in range(len(segment) - 1): p1, p2 = segment[i], segment[i+1]; length += math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return length

def extract_paths_from_skeleton(skeleton):
    points = set((y, x) for y, x in zip(*np.nonzero(skeleton)))
    paths = []
    while points:
        start_node = points.pop()
        
        # 순방향 탐색
        forward = []
        curr = start_node
        while True:
            y, x = curr
            found = None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0: continue
                    nxt = (y + dy, x + dx)
                    if nxt in points:
                        found = nxt
                        break
                if found: break
            if found:
                forward.append(found)
                points.remove(found)
                curr = found
            else:
                break
                
        # 역방향 탐색
        backward = []
        curr = start_node
        while True:
            y, x = curr
            found = None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0: continue
                    nxt = (y + dy, x + dx)
                    if nxt in points:
                        found = nxt
                        break
                if found: break
            if found:
                backward.append(found)
                points.remove(found)
                curr = found
            else:
                break
                
        full_path = backward[::-1] + [start_node] + forward
        paths.append(full_path)
    return paths

def vectorize_and_sort_paths(binary_image, line_epsilon, min_points=3, filter_small_loops=False, scale_factor=1.0, offset=(0, 0)):
    from skimage.morphology import skeletonize
    # 입력 마스크를 항상 1픽셀 중심선으로 확실하게 뼈대화하여 두꺼운 선에 의한 지그재그 경로 및 과도한 분할 방지
    skeleton = skeletonize(binary_image > 0)
    raw_paths = extract_paths_from_skeleton(skeleton)
    vectorized_paths = []
    x_offset, y_offset = offset
    
    for path in raw_paths:
        if len(path) < min_points:
            continue
            
        # 캔버스 실제 픽셀 좌표계로 먼저 변환하여 근사를 수행
        # 이를 통해 Line Epsilon (px)이 분석 해상도나 축소율에 왜곡되지 않고 실제 캔버스 화면 픽셀 기준으로 1:1 정확히 계산됨
        if scale_factor != 1.0 or offset != (0, 0):
            canvas_pts = [[x_offset + x * scale_factor, y_offset + y * scale_factor] for y, x in path]
        else:
            canvas_pts = [[float(x), float(y)] for y, x in path]
            
        cnt = np.array([[[pt[0], pt[1]]] for pt in canvas_pts], dtype=np.float32)
        if line_epsilon <= 0:
            line_segment = [tuple(p[0]) for p in cnt]
        else:
            approx = cv2.approxPolyDP(cnt, line_epsilon, False)
            line_segment = [tuple(p[0]) for p in approx]
            
        # 유효한 선분은 최소 2개의 점(시작점, 끝점)으로 구성되므로 >= 2 조건으로 보존
        # 기존 min_points=4 조건으로 인해 직선/완만한 곡선이 통째로 삭제되던 버그 완벽 해결
        if len(line_segment) >= 2:
            seg_len = calculate_segment_length(line_segment)
            if seg_len < 3.0:
                continue # 3픽셀 미만의 미세 고립 잡점만 제거
                
            if filter_small_loops and len(line_segment) >= 3:
                p_start, p_end = line_segment[0], line_segment[-1]
                is_closed = (p_start[0] - p_end[0])**2 + (p_start[1] - p_end[1])**2 <= 25
                if is_closed and seg_len < 35.0:
                    continue  # 지저분한 미세 다각형/얼룩 노이즈 제거
                    
            vectorized_paths.append(line_segment)
            
    if not vectorized_paths:
        return []
        
    # 가장 긴 라인 순서대로 그리도록 길이 기준 내림차순 정렬
    vectorized_paths.sort(key=calculate_segment_length, reverse=True)
    
    def distance(p1, p2): return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    sorted_paths = [vectorized_paths[0]]
    for p in vectorized_paths[1:]:
        last_point = sorted_paths[-1][-1]
        d1 = distance(last_point, p[0])
        d2 = distance(last_point, p[-1])
        if d2 < d1:
            p.reverse()
        sorted_paths.append(p)
        
    return sorted_paths

def align_plan_to_center(line_drawing_plan, canvas_width, canvas_height, logger=None):
    if not line_drawing_plan:
        return line_drawing_plan
        
    all_x = [pt[0] for seg in line_drawing_plan for pt in seg]
    all_y = [pt[1] for seg in line_drawing_plan for pt in seg]
    
    if not all_x or not all_y:
        return line_drawing_plan
        
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    content_w = max_x - min_x
    content_h = max_y - min_y
    
    target_center_x = canvas_width / 2.0
    target_center_y = canvas_height / 2.0
    current_center_x = (min_x + max_x) / 2.0
    current_center_y = (min_y + max_y) / 2.0
    
    offset_x = int(round(target_center_x - current_center_x))
    offset_y = int(round(target_center_y - current_center_y))
    
    if logger and (offset_x != 0 or offset_y != 0):
        logger.info(f"선화 중앙 정렬 보정: X 오프셋={offset_x}, Y 오프셋={offset_y} (내용: {content_w}x{content_h}, 캔버스: {canvas_width}x{canvas_height})")
        
    centered_plan = [[(x + offset_x, y + offset_y) for x, y in seg] for seg in line_drawing_plan]
    return centered_plan

def generate_plan_from_image(final_image, logger, line_epsilon, contour_mode, contour_method, uses_lineart_model=False):
    logger.info("수정된 이미지로부터 그리기 계획을 생성합니다...")
    gray_image = cv2.cvtColor(np.array(final_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
    
    # 여기서 한 번 더 뼈대화(Skeletonize)를 거쳐 이중선(외곽선 따기) 문제 완벽 방지
    # 사용자가 연필로 수정한 두꺼운 선이나, 미리보기용으로 생성된 2px 선의 중심만 정확히 1픽셀로 추출
    from skimage.morphology import skeletonize
    skeleton = skeletonize(binary_image > 0)
    binary_image = (skeleton * 255).astype(np.uint8)
    
    # DFS를 사용한 진짜 벡터화 및 TSP를 이용한 최단 이동거리 정렬 수행 (min_points=3으로 직선 보존)
    line_drawing_plan = vectorize_and_sort_paths(binary_image, line_epsilon, min_points=3)
    line_drawing_plan = [[(int(round(x)), int(round(y))) for x, y in seg] for seg in line_drawing_plan]
    logger.info(f"벡터화 및 TSP 정렬 완료: {len(line_drawing_plan)}개의 최적화된 경로 생성.")
    
    return line_drawing_plan

def generate_preview_image(image_path, pipeline, combination_method, canvas_coords, precision, logger, line_epsilon, contour_mode, contour_method, use_denoise_filter=True, hatch_mode="외곽선 + 빗금", hatch_pattern="45° (대각선)", hatch_spacing=8, hatch_adaptive=True):
    logger.info(f"이미지 처리 시작 (조합 방식: {combination_method}, 노이즈 완화 필터: {'On' if use_denoise_filter else 'Off'})...")
    if not image_path: return None, None
    user_image, canvas_width, canvas_height = Image.open(image_path).convert("RGB"), canvas_coords[2]-canvas_coords[0], canvas_coords[3]-canvas_coords[1]
    img_width, img_height = user_image.size
    ratio = min(canvas_width / img_width, canvas_height / img_height) if img_width > 0 and img_height > 0 else 0
    target_w, target_h = int(img_width * ratio), int(img_height * ratio)
    precision_scale = precision / 100.0
    analysis_w, analysis_h = max(1, int(target_w * precision_scale)), max(1, int(target_h * precision_scale))
    image_for_analysis = user_image.resize((analysis_w, analysis_h), Image.Resampling.LANCZOS)
    logger.info(f"이미지를 분석용 크기 {analysis_w}x{analysis_h}로 리사이즈.")
    
    # 음영 노이즈 완화 필터: 엣지는 보존하고 부드러운 그라데이션 및 명암을 매끄럽게 평활화하여 다각형 발생 원천 차단
    if use_denoise_filter:
        logger.info("음영 노이즈 완화 필터(Bilateral Filter) 적용 중...")
        img_np = cv2.cvtColor(np.array(image_for_analysis), cv2.COLOR_RGB2BGR)
        img_np = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
        image_for_analysis = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    
    uses_lineart_model = any(is_lineart_model(layer["model"]) for layer in pipeline)
    
    combined_edges = np.zeros((analysis_h, analysis_w), dtype=np.uint8)
    first_layer = True
    
    for layer in pipeline:
        edges = get_edges_from_model(layer["model"], image_for_analysis, layer["threshold"], logger)
        if edges is not None:
            if first_layer:
                combined_edges = edges
                first_layer = False
            else:
                if "교집합" in combination_method or "Intersection" in combination_method:
                    combined_edges = cv2.bitwise_and(combined_edges, edges)
                elif "차집합" in combination_method or "Difference" in combination_method:
                    combined_edges = cv2.bitwise_and(combined_edges, cv2.bitwise_not(edges))
                else:
                    # "합 연산 (Union)", "Union (Combine)", "오버레이 (Overlay)", "Overlay" 등 모든 병합 방식은 다중 레이어를 합성
                    combined_edges = cv2.bitwise_or(combined_edges, edges)
                
    # 면(채색 영역 및 굵은 선)에 대해 설정된 모드(외곽선+빗금, 스켈레톤, 빗금만) 및 패턴 적용
    logger.info(f"면 처리 모드 적용 중: {hatch_mode} (패턴: {hatch_pattern}, 간격: {hatch_spacing}px, 적응형: {'On' if hatch_adaptive else 'Off'})...")
    combined_edges = process_surface_with_hatching(
        combined_edges,
        orig_image=image_for_analysis,
        mode=hatch_mode,
        pattern=hatch_pattern,
        spacing=int(hatch_spacing),
        adaptive=bool(hatch_adaptive)
    )
        
    scale_factor = target_w / analysis_w if analysis_w > 0 else 1.0
    x_offset, y_offset = (canvas_width - target_w) // 2, (canvas_height - target_h) // 2

    # DFS를 사용한 진짜 벡터화 및 TSP를 이용한 최단 이동거리 정렬 수행 (필터 켜짐 시 미세 루프 제거)
    # 캔버스 실제 픽셀 좌표계로 먼저 변환한 후 approxPolyDP 근사를 수행하여 line_epsilon이 화면 픽셀 기준으로 1:1 정확히 계산됨
    vectorized_plan = vectorize_and_sort_paths(
        combined_edges, 
        line_epsilon=line_epsilon, 
        min_points=3, 
        filter_small_loops=use_denoise_filter,
        scale_factor=scale_factor,
        offset=(x_offset, y_offset)
    )
    logger.info(f"벡터화 및 TSP 정렬 완료: {len(vectorized_plan)}개의 최적화된 경로 생성.")

    # 선화 내용물을 캔버스 좌/우 및 위/아래 중앙으로 완벽 정렬
    final_plan = align_plan_to_center(vectorized_plan, canvas_width, canvas_height, logger)
    final_plan = [[(int(round(x)), int(round(y))) for x, y in seg] for seg in final_plan]

    preview_image = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    preview_draw = ImageDraw.Draw(preview_image)
    for segment in final_plan:
        if len(segment) >= 2:
            preview_draw.line(segment, fill=(0, 0, 0), width=1)
        elif len(segment) == 1:
            preview_draw.point(segment[0], fill=(0, 0, 0))
    
    # 캔버스 크기(canvas_width, canvas_height)를 그대로 유지하여 편집기 및 실제 그리기 좌표계와 일치
    return preview_image, final_plan

class AutoDrawerLine:
    def __init__(self, line_drawing_plan, canvas_area_str, update_cb, stop_event, logger, line_delay=0.001, mouse_duration=0, moves_per_second=0):
        self.canvas_area = tuple(map(int, canvas_area_str.split(',')))
        canvas_width = self.canvas_area[2] - self.canvas_area[0]
        canvas_height = self.canvas_area[3] - self.canvas_area[1]
        
        # 지정된 캔버스 영역의 좌/우 및 위/아래 중앙 정렬 보정
        self.line_drawing_plan = align_plan_to_center(line_drawing_plan, canvas_width, canvas_height, logger)
        
        self.update_callback, self.stop_event, self.logger, self.line_delay, self.mouse_duration = \
            update_cb, stop_event, logger, line_delay, mouse_duration
        self.moves_per_second = float(moves_per_second) if moves_per_second else 0.0
    def run(self):
        pyautogui.FAILSAFE = False
        original_pause = pyautogui.PAUSE
        pyautogui.PAUSE = 0  # pyautogui 내부 0.1초 강제 대기 오버라이드
        rate_limit_info = f"{self.moves_per_second:g}회/초" if self.moves_per_second > 0 else "무제한"
        self.logger.info(f"자동 그리기 시작... (내부 지연 오버라이드 적용, 초당 이동 제한: {rate_limit_info})")
        total_lines = len(self.line_drawing_plan)
        last_update_time = time.time()

        try:
            # 시작점으로 지연 없이 즉시 이동
            pyautogui.moveTo(self.canvas_area[0] + self.line_drawing_plan[0][0][0], self.canvas_area[1] + self.line_drawing_plan[0][0][1], duration=0, _pause=False)

            for i, line_segment in enumerate(self.line_drawing_plan):
                if self.stop_event.is_set(): break
                if len(line_segment) < 2: continue

                # GUI 업데이트 최적화: 0.05초마다 또는 마지막에만 업데이트
                current_time = time.time()
                if (current_time - last_update_time > 0.05) or (i == total_lines - 1):
                    current_line_num = i + 1; progress_text = f"{current_line_num}/{total_lines}"
                    line_progress = (current_line_num / total_lines) * 100
                    self.update_callback(f"선 {progress_text}", line_progress, progress_text, line_progress)
                    last_update_time = current_time

                first_rel = line_segment[0]
                abs_x0, abs_y0 = self.canvas_area[0] + first_rel[0], self.canvas_area[1] + first_rel[1]
                pyautogui.moveTo(abs_x0, abs_y0, duration=0, _pause=False) # 시작점으로 즉시 이동
                pyautogui.mouseDown(_pause=False)
                for point_rel in line_segment[1:]:
                    if self.stop_event.is_set(): break
                    abs_x, abs_y = self.canvas_area[0] + point_rel[0], self.canvas_area[1] + point_rel[1]
                    move_start = time.perf_counter()
                    pyautogui.moveTo(abs_x, abs_y, duration=self.mouse_duration, _pause=False)
                    if self.moves_per_second > 0:
                        target_interval = 1.0 / self.moves_per_second
                        elapsed = time.perf_counter() - move_start
                        remaining = target_interval - elapsed
                        if remaining > 0:
                            time.sleep(remaining)
                pyautogui.mouseUp(_pause=False)
                if self.line_delay > 0:
                    time.sleep(self.line_delay)
            self.logger.info("그리기 완료!"); self.update_callback("완료", 100, "완료", 100)
        finally:
            pyautogui.PAUSE = original_pause
            pyautogui.FAILSAFE = True

def calculate_plan_time_estimate(line_drawing_plan, line_delay=0.01, mouse_duration=0.0, moves_per_second=0.0):
    """그리기 계획에 대한 예상 소요 시간(초), 유효 선 개수, 총 점 개수 계산"""
    if not line_drawing_plan:
        return 0.0, 0, 0
    valid_lines = [seg for seg in line_drawing_plan if len(seg) >= 2]
    total_lines = len(valid_lines)
    total_points = sum(len(seg) for seg in valid_lines)
    total_draw_points = sum(len(seg) - 1 for seg in valid_lines)
    
    if moves_per_second > 0:
        point_interval = max(1.0 / moves_per_second, mouse_duration)
    else:
        point_interval = mouse_duration if mouse_duration > 0 else 0.0003
        
    points_time = total_draw_points * point_interval
    lines_time = total_lines * (line_delay + 0.001)
    
    total_seconds = points_time + lines_time
    return total_seconds, total_lines, total_points

def format_duration(seconds):
    """초 단위 시간을 사람이 읽기 쉬운 형식('약 X분 Y초')으로 포맷팅"""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}초"
    minutes = seconds // 60
    rem_seconds = seconds % 60
    if rem_seconds == 0:
        return f"{minutes}분"
    return f"{minutes}분 {rem_seconds}초"

class EditorWindow(Toplevel):
    def __init__(self, root, initial_image, line_epsilon, contour_mode, contour_method, start_callback, cancel_callback, uses_lineart_model=False, line_delay=0.01, mouse_duration=0.0, moves_per_second=0.0, initial_plan=None):
        super().__init__(root)
        self.title("편집기"); self.start_callback, self.cancel_callback = start_callback, cancel_callback
        self.line_epsilon, self.contour_mode, self.contour_method = line_epsilon, contour_mode, contour_method
        self.uses_lineart_model = uses_lineart_model
        self.line_delay = line_delay
        self.mouse_duration = mouse_duration
        self.moves_per_second = moves_per_second
        self.initial_plan = initial_plan
        self.is_modified = False
        self.image = initial_image.copy(); self.img_w, self.img_h = self.image.size
        
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        max_w, max_h = int(screen_w * 0.9), int(screen_h * 0.9)
        chrome_h = 160
        win_w, win_h = min(self.img_w + 40, max_w), min(self.img_h + chrome_h, max_h)
        self.geometry(f"{win_w}x{win_h}")

        # --- 상단 정보 바 (그리기 완료 예상 시간 및 통계) ---
        info_frame = tk.Frame(self, bg="#1e293b", bd=1, relief=tk.SOLID)
        info_frame.pack(fill=tk.X, padx=10, pady=(6, 2))
        self.time_info_label = tk.Label(
            info_frame, 
            text="⏳ 그리기 완료 예상 시간 계산 중...", 
            font=("맑은 고딕", 10, "bold"), 
            bg="#1e293b", 
            fg="#38bdf8", 
            pady=6
        )
        self.time_info_label.pack()

        self.tool, self.eraser_size, self.last_pos = "pencil", 5, None
        self.canvas = Canvas(self, bg="white", highlightthickness=0); self.canvas.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor="center")
        self.eraser_cursor = self.canvas.create_oval(0,0,0,0, outline="gray", width=1, state='hidden')
        
        btn_frame = tk.Frame(self); btn_frame.pack(pady=(0, 5), fill=tk.X, padx=10)
        tk.Button(btn_frame, text="진행", command=self.on_start).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        tk.Button(btn_frame, text="취소", command=self.on_cancel).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        tool_frame = tk.Frame(self); tool_frame.pack(pady=(0, 8), fill=tk.X, padx=10)
        tk.Button(tool_frame, text="연필", command=self.select_pencil).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        tk.Button(tool_frame, text="지우개", command=self.select_eraser).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        self.canvas.bind("<Configure>", self.redraw_canvas); self.canvas.bind("<ButtonPress-1>", self.on_press); self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<B1-Motion>", self.on_drag); self.canvas.bind("<MouseWheel>", self.on_wheel); self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", lambda e: self.canvas.itemconfig(self.eraser_cursor, state='hidden'))
        
        self.select_pencil(); self.transient(root); self.grab_set()
        self.after(50, self.update_time_estimate)

    def update_time_estimate(self):
        try:
            if not self.is_modified and self.initial_plan is not None:
                plan = self.initial_plan
            else:
                plan = generate_plan_from_image(
                    self.image, logging.getLogger(), self.line_epsilon, 
                    self.contour_mode, self.contour_method, uses_lineart_model=self.uses_lineart_model
                )
            est_seconds, total_lines, total_points = calculate_plan_time_estimate(
                plan, self.line_delay, self.mouse_duration, self.moves_per_second
            )
            time_str = format_duration(est_seconds)
            rate_info = f"({self.moves_per_second:g}회/초 기준)" if self.moves_per_second > 0 else "(무제한 속도 기준)"
            self.time_info_label.config(
                text=f"⏱️ 그리기 완료 예상 시간: 약 {time_str} {rate_info}  |  총 선: {total_lines:,}개  |  포인트: {total_points:,}개",
                fg="#38bdf8"
            )
        except Exception as e:
            self.time_info_label.config(text=f"예상 시간 계산 실패: {e}", fg="#f87171")

    def _get_display_geometry(self):
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if self.img_w <= 1 or self.img_h <= 1 or canvas_w <= 1 or canvas_h <=1: return 0, 0, 0, 0, 0
        scale = min(canvas_w / self.img_w, canvas_h / self.img_h)
        disp_w, disp_h = int(self.img_w * scale), int(self.img_h * scale)
        offset_x, offset_y = (canvas_w - disp_w) // 2, (canvas_h - disp_h) // 2
        return scale, offset_x, offset_y, disp_w, disp_h

    def select_pencil(self): 
        self.tool = "pencil"
        self.canvas.config(cursor="cross")
        self.canvas.itemconfig(self.eraser_cursor, state='hidden')
        
    def select_eraser(self): 
        self.tool = "eraser"
        self.canvas.config(cursor="none")
        self.canvas.itemconfig(self.eraser_cursor, state='normal')
        self.update_eraser_cursor(0,0)

    def on_press(self, event):
        self.last_pos = (event.x, event.y)
        if self.tool == "eraser": self.canvas.itemconfig(self.eraser_cursor, fill="gray", stipple="gray50")
        self.on_drag(event)
        
    def on_release(self, event): 
        self.last_pos = None
        if self.tool == "eraser": self.canvas.itemconfig(self.eraser_cursor, fill="", stipple="")
        self.after(50, self.update_time_estimate)
        
    def on_wheel(self, event):
        if self.tool == "eraser": self.eraser_size += 1 if event.delta > 0 else -1; self.eraser_size = max(1, self.eraser_size); self.update_eraser_cursor(event.x, event.y)
    
    def on_mouse_move(self, event):
        if self.tool == "eraser": self.canvas.itemconfig(self.eraser_cursor, state='normal'); self.update_eraser_cursor(event.x, event.y)

    def on_drag(self, event):
        if self.tool == "eraser": self.update_eraser_cursor(event.x, event.y)
        if self.last_pos:
            scale, offset_x, offset_y, _, _ = self._get_display_geometry()
            if scale == 0: return
            x, y = event.x, event.y
            img_x1 = (self.last_pos[0] - offset_x) / scale; img_y1 = (self.last_pos[1] - offset_y) / scale
            img_x2 = (x - offset_x) / scale; img_y2 = (y - offset_y) / scale
            draw = ImageDraw.Draw(self.image)
            if self.tool == "pencil": draw.line([(img_x1, img_y1), (img_x2, img_y2)], fill="black", width=2)
            elif self.tool == "eraser": es = self.eraser_size / scale; draw.ellipse([(img_x2-es, img_y2-es), (img_x2+es, img_y2+es)], fill="white")
            self.is_modified = True
            self.last_pos = (x, y)
            self.redraw_canvas()

    def update_eraser_cursor(self, x, y): s = self.eraser_size; self.canvas.coords(self.eraser_cursor, x-s, y-s, x+s, y+s)
    
    def redraw_canvas(self, event=None):
        if not hasattr(self, 'canvas') or not self.canvas.winfo_exists(): return
        
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1: return
            
        scale, offset_x, offset_y, new_w, new_h = self._get_display_geometry()
        if scale == 0: return

        disp_img = self.image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(disp_img)
        
        # --- 수정된 부분: 이미지 위치를 캔버스 중앙으로 설정 ---
        self.canvas.itemconfig(self.image_on_canvas, image=self.photo_image)
        self.canvas.coords(self.image_on_canvas, canvas_w/2, canvas_h/2) # 앵커가 center이므로 중앙 좌표 사용
        
        self.canvas.tag_raise(self.eraser_cursor)
        
    def on_start(self):
        # 크롭된 이미지가 아닌 원본 캔버스 크기 기준으로 plan 생성
        # 편집기에서 수정한 내용이 없다면 초기 미리보기 계산 계획을 100% 그대로 전달 (재벡터화/선 손실 방지)
        if not self.is_modified and self.initial_plan is not None:
            final_plan = self.initial_plan
        else:
            final_plan = generate_plan_from_image(self.image, logging.getLogger(), self.line_epsilon, self.contour_mode, self.contour_method, uses_lineart_model=self.uses_lineart_model)
        self.start_callback(final_plan); self.destroy()
        
    def on_cancel(self): self.cancel_callback(); self.destroy()
