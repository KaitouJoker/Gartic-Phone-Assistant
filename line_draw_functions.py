import tkinter as tk
from tkinter import Toplevel, Canvas
from PIL import Image, ImageTk, ImageGrab, ImageDraw, ImageOps
import pyautogui
import time
import threading
import numpy as np
import cv2
from pynput import keyboard
import math
import logging

import torch
from controlnet_aux import HEDdetector, PidiNetDetector, LineartAnimeDetector

model_cache = {}
CONTOUR_MODES = {"외곽선만 찾기": cv2.RETR_EXTERNAL, "모든 선 찾기": cv2.RETR_LIST, "모든 선 찾기 + 계층": cv2.RETR_TREE}
CONTOUR_METHODS = {"선 압축하기": cv2.CHAIN_APPROX_SIMPLE, "모든 점 저장하기": cv2.CHAIN_APPROX_NONE}

def load_processor_model(model_name, logger):
    if model_name in model_cache: logger.info(f"캐시에서 {model_name} 모델을 로드했습니다."); return model_cache[model_name]
    logger.info(f"{model_name} 모델 로딩 중... (첫 실행 시 시간이 걸릴 수 있습니다)")
    model = None
    try:
        if model_name == "HED": model = HEDdetector.from_pretrained("lllyasviel/Annotators")
        elif model_name == "SoftEdge": model = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        elif model_name == "Lineart Anime": model = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
        if model: logger.info(f"{model_name} 모델 로딩 완료."); model_cache[model_name] = model
        return model
    except Exception as e: logger.exception(f"{model_name} 모델 로딩 실패: {e}"); return None

def load_settings():
    settings = {};
    try:
        with open("line_draw_setting.txt", "r") as f:
            for line in f: key, value = line.strip().split("=", 1); settings[key] = value
    except FileNotFoundError: return {}
    return settings

def save_settings(app):
    settings = {"CANVAS_AREA": app.canvas_area_var.get(), "IMAGE_PATH": app.image_path_var.get(), "PRECISION": app.precision_var.get(), "LINE_EPSILON": app.line_epsilon_var.get(), "LINE_DELAY": app.line_delay_var.get(), "MOUSE_DURATION": app.mouse_duration_var.get(), "CONTOUR_MODE": app.contour_mode_var.get(), "CONTOUR_METHOD": app.contour_method_var.get(), "COMBINATION_METHOD": app.combination_method_var.get(), "NUM_LAYERS": app.num_layers_var.get()}
    for i, layer in enumerate(app.layers):
        settings[f"L{i+1}_ENABLED"] = layer["enabled"].get(); settings[f"L{i+1}_MODEL"] = layer["model"].get(); settings[f"L{i+1}_THRESHOLD"] = layer["threshold"].get()
    with open("line_draw_setting.txt", "w") as f:
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

def get_edges_from_model(model_name, image, threshold, logger):
    edges = None; target_size = image.size
    if model_name == "Canny":
        logger.info(f"Canny 엣지 검출 (Threshold: {threshold})...")
        edges = cv2.Canny(cv2.GaussianBlur(np.array(image.convert("L")), (5, 5), 0), int(255*threshold/2), int(255*threshold))
    else:
        model = load_processor_model(model_name, logger)
        if model:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
            logger.info(f"{model_name} 모델 추론 (Threshold: {threshold})...")
            with torch.no_grad(): edge_map = np.array(model(image, safe_steps=True))
            edges = (edge_map > int(255 * threshold)).astype(np.uint8) * 255
            if len(edges.shape) == 3: edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    if edges is not None:
        if edges.shape[1] != target_size[0] or edges.shape[0] != target_size[1]:
            logger.warning(f"{model_name} 모델 출력 크기({edges.shape[1]}x{edges.shape[0]})가 분석 크기와 다릅니다. 강제 조정합니다.")
            edges = cv2.resize(edges, target_size, interpolation=cv2.INTER_AREA)
    return edges

def calculate_segment_length(segment):
    length = 0.0
    for i in range(len(segment) - 1): p1, p2 = segment[i], segment[i+1]; length += math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return length

def generate_plan_from_image(final_image, logger, line_epsilon, contour_mode, contour_method):
    logger.info("수정된 이미지로부터 그리기 계획을 생성합니다...")
    gray_image = cv2.cvtColor(np.array(final_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
    mode, method = CONTOUR_MODES.get(contour_mode, cv2.RETR_EXTERNAL), CONTOUR_METHODS.get(contour_method, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(binary_image, mode, method)
    logger.info(f"편집된 이미지에서 {len(contours)}개의 컨투어 발견.")
    line_drawing_plan = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 4: continue
        approx = cv2.approxPolyDP(cnt, line_epsilon, True)
        line_segment = [tuple(point[0]) for point in approx]
        if len(line_segment) > 1: line_drawing_plan.append(line_segment)
    logger.info("그리기 순서를 선 길이에 따라 정렬합니다 (긴 선 먼저)...")
    line_drawing_plan.sort(key=calculate_segment_length, reverse=True)
    return line_drawing_plan

def generate_preview_image(image_path, pipeline, combination_method, canvas_coords, precision, logger, line_epsilon, contour_mode, contour_method):
    logger.info(f"이미지 처리 시작 (조합 방식: {combination_method})...")
    if not image_path: return None
    user_image, canvas_width, canvas_height = Image.open(image_path).convert("RGB"), canvas_coords[2]-canvas_coords[0], canvas_coords[3]-canvas_coords[1]
    img_width, img_height = user_image.size
    ratio = min(canvas_width / img_width, canvas_height / img_height) if img_width > 0 and img_height > 0 else 0
    target_w, target_h = int(img_width * ratio), int(img_height * ratio)
    precision_scale = precision / 100.0
    analysis_w, analysis_h = max(1, int(target_w * precision_scale)), max(1, int(target_h * precision_scale))
    image_for_analysis = user_image.resize((analysis_w, analysis_h), Image.LANCZOS)
    logger.info(f"이미지를 분석용 크기 {analysis_w}x{analysis_h}로 리사이즈.")
    mode, method = CONTOUR_MODES.get(contour_mode, cv2.RETR_EXTERNAL), CONTOUR_METHODS.get(contour_method, cv2.CHAIN_APPROX_SIMPLE)
    final_contours = []
    if combination_method == "Union (Combine)":
        combined_edges = np.zeros((analysis_h, analysis_w), dtype=np.uint8)
        for layer in pipeline:
            edges = get_edges_from_model(layer["model"], image_for_analysis, layer["threshold"], logger)
            if edges is not None: combined_edges = cv2.bitwise_or(combined_edges, edges)
        contours, _ = cv2.findContours(combined_edges, mode, method); final_contours.extend(contours)
    else: # Overlay
        for layer in pipeline:
            edges = get_edges_from_model(layer["model"], image_for_analysis, layer["threshold"], logger)
            if edges is not None:
                contours, _ = cv2.findContours(edges, mode, method); final_contours.extend(contours)
    logger.info(f"총 {len(final_contours)}개의 최종 컨투어 발견.")
    preview_image = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255)); preview_draw = ImageDraw.Draw(preview_image)
    scale_factor = target_w / analysis_w if analysis_w > 0 else 0
    x_offset, y_offset = (canvas_width - target_w) // 2, (canvas_height - target_h) // 2
    temp_plan = []
    for cnt in final_contours:
        if cv2.contourArea(cnt) < 20: continue
        approx = cv2.approxPolyDP(cnt, line_epsilon, True)
        line_segment = [ (x_offset + int(p[0][0] * scale_factor), y_offset + int(p[0][1] * scale_factor)) for p in approx ]
        if len(line_segment) > 1: temp_plan.append(line_segment)
    temp_plan.sort(key=calculate_segment_length, reverse=True)
    for segment in temp_plan: preview_draw.line(segment, fill=(0, 0, 0), width=1)
    return crop_image_to_content(preview_image, logger)

class AutoDrawerLine:
    def __init__(self, line_drawing_plan, canvas_area_str, update_cb, stop_event, logger, line_delay=0.01, mouse_duration=0.005):
        self.line_drawing_plan, self.canvas_area, self.update_callback, self.stop_event, self.logger, self.line_delay, self.mouse_duration = \
            line_drawing_plan, tuple(map(int, canvas_area_str.split(','))), update_cb, stop_event, logger, line_delay, mouse_duration
    def run(self):
        pyautogui.FAILSAFE = False; self.logger.info("자동 그리기 시작...")
        total_lines = len(self.line_drawing_plan)
        for i, line_segment in enumerate(self.line_drawing_plan):
            if self.stop_event.is_set(): break
            if len(line_segment) < 2: continue
            current_line_num = i + 1; progress_text = f"{current_line_num}/{total_lines}"
            line_progress = (current_line_num / total_lines) * 100
            self.update_callback(f"선 {progress_text}", line_progress, progress_text, line_progress)
            first_rel = line_segment[0]
            abs_x0, abs_y0 = self.canvas_area[0] + first_rel[0], self.canvas_area[1] + first_rel[1]
            pyautogui.moveTo(abs_x0, abs_y0, duration=0.01); time.sleep(0.01)
            pyautogui.mouseDown()
            for point_rel in line_segment[1:]:
                if self.stop_event.is_set(): break
                abs_x, abs_y = self.canvas_area[0] + point_rel[0], self.canvas_area[1] + point_rel[1]
                pyautogui.moveTo(abs_x, abs_y, duration=self.mouse_duration)
            pyautogui.mouseUp(); time.sleep(self.line_delay)
        self.logger.info("그리기 완료!"); self.update_callback("완료", 100, "완료", 100); pyautogui.FAILSAFE = True

class EditorWindow(Toplevel):
    def __init__(self, root, initial_image, line_epsilon, contour_mode, contour_method, start_callback, cancel_callback):
        super().__init__(root)
        self.title("편집기"); self.start_callback, self.cancel_callback = start_callback, cancel_callback
        self.line_epsilon, self.contour_mode, self.contour_method = line_epsilon, contour_mode, contour_method
        self.image = initial_image.copy(); self.img_w, self.img_h = self.image.size
        
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        max_w, max_h = int(screen_w * 0.9), int(screen_h * 0.9)
        chrome_h = 120
        win_w, win_h = min(self.img_w + 40, max_w), min(self.img_h + chrome_h, max_h)
        self.geometry(f"{win_w}x{win_h}")

        self.tool, self.eraser_size, self.last_pos = "pencil", 5, None
        self.canvas = Canvas(self, bg="white", highlightthickness=0); self.canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor="center")
        self.eraser_cursor = self.canvas.create_oval(0,0,0,0, outline="gray", width=1, state='hidden')
        
        btn_frame = tk.Frame(self); btn_frame.pack(pady=(0, 5), fill=tk.X, padx=10)
        tk.Button(btn_frame, text="진행", command=self.on_start).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        tk.Button(btn_frame, text="취소", command=self.on_cancel).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        tool_frame = tk.Frame(self); tool_frame.pack(pady=(0, 10), fill=tk.X, padx=10)
        tk.Button(tool_frame, text="연필", command=self.select_pencil).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        tk.Button(tool_frame, text="지우개", command=self.select_eraser).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        self.canvas.bind("<Configure>", self.redraw_canvas); self.canvas.bind("<ButtonPress-1>", self.on_press); self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<B1-Motion>", self.on_drag); self.canvas.bind("<MouseWheel>", self.on_wheel); self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", lambda e: self.canvas.itemconfig(self.eraser_cursor, state='hidden'))
        
        self.select_pencil(); self.transient(root); self.grab_set()

    def _get_display_geometry(self):
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if self.img_w <= 1 or self.img_h <= 1 or canvas_w <= 1 or canvas_h <=1: return 0, 0, 0, 0, 0
        scale = min(canvas_w / self.img_w, canvas_h / self.img_h)
        disp_w, disp_h = int(self.img_w * scale), int(self.img_h * scale)
        offset_x, offset_y = (canvas_w - disp_w) // 2, (canvas_h - disp_h) // 2
        return scale, offset_x, offset_y, disp_w, disp_h

    def select_pencil(self): self.tool = "pencil"; self.canvas.config(cursor="cross"); self.canvas.itemconfig(self.eraser_cursor, state='hidden')
    def select_eraser(self): self.tool = "eraser"; self.canvas.config(cursor="none"); self.canvas.itemconfig(self.eraser_cursor, state='normal'); self.update_eraser_cursor(0,0)

    def on_press(self, event):
        self.last_pos = (event.x, event.y)
        if self.tool == "eraser": self.canvas.itemconfig(self.eraser_cursor, fill="gray", stipple="gray50")
        self.on_drag(event)
        
    def on_release(self, event): 
        self.last_pos = None
        if self.tool == "eraser": self.canvas.itemconfig(self.eraser_cursor, fill="", stipple="")
        
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
            self.last_pos = (x, y)
            self.redraw_canvas()

    def update_eraser_cursor(self, x, y): s = self.eraser_size; self.canvas.coords(self.eraser_cursor, x-s, y-s, x+s, y+s)
    
    def redraw_canvas(self, event=None):
        if not hasattr(self, 'canvas') or not self.canvas.winfo_exists(): return
        
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1: return
            
        scale, offset_x, offset_y, new_w, new_h = self._get_display_geometry()
        if scale == 0: return

        disp_img = self.image.resize((new_w, new_h), Image.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(disp_img)
        
        # --- 수정된 부분: 이미지 위치를 캔버스 중앙으로 설정 ---
        self.canvas.itemconfig(self.image_on_canvas, image=self.photo_image)
        self.canvas.coords(self.image_on_canvas, canvas_w/2, canvas_h/2) # 앵커가 center이므로 중앙 좌표 사용
        
        self.canvas.tag_raise(self.eraser_cursor)
        
    def on_start(self):
        # 크롭된 이미지가 아닌 원본 캔버스 크기 기준으로 plan 생성
        final_plan = generate_plan_from_image(self.image, logging.getLogger(), self.line_epsilon, self.contour_mode, self.contour_method)
        self.start_callback(final_plan); self.destroy()
        
    def on_cancel(self): self.cancel_callback(); self.destroy()