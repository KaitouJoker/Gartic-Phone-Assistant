import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
import threading
import logging
import time
from pynput import keyboard

import line_draw_functions as fn

stop_drawing_flag = threading.Event()
drawing_thread = None

class CtkTextboxHandler(logging.Handler):
    def __init__(self, textbox):
        super().__init__()
        self.textbox = textbox
    def emit(self, record):
        msg = self.format(record)
        def append_message():
            self.textbox.configure(state="normal")
            self.textbox.insert(tk.END, msg + "\n")
            self.textbox.configure(state="disabled")
            self.textbox.see(tk.END)
        self.textbox.after(0, append_message)

class LineDrawApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Gartic Phone Line Drawer v2.4 (Editor Fix)")
        self.geometry("500x950")
        self.grid_columnconfigure(0, weight=1)

        # --- 변수 선언 ---
        self.canvas_area_var = ctk.StringVar()
        self.image_path_var = ctk.StringVar()
        self.precision_var = ctk.StringVar(value="100%")
        self.line_epsilon_var = ctk.StringVar(value="2.0")
        self.line_delay_var = ctk.StringVar(value="0.01")
        self.mouse_duration_var = ctk.DoubleVar(value=0.0001)
        self.contour_mode_var = ctk.StringVar(value="모든 선 찾기")
        self.contour_method_var = ctk.StringVar(value="선 압축하기")
        
        self.num_layers_var = ctk.StringVar(value="1")
        self.layers = []
        self.combination_method_var = ctk.StringVar(value="Union (Combine)")

        self.total_progress_var = ctk.DoubleVar(value=0.0)
        self.total_status_var = ctk.StringVar(value="대기")
        self.preview_image = None
        self.drawing_plan = []

        # --- GUI 위젯 ---
        row_idx = 0
        
        selection_frame = ctk.CTkFrame(self)
        selection_frame.grid(row=row_idx, column=0, padx=10, pady=10, sticky="ew")
        selection_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(selection_frame, text="1. 캔버스 영역", command=lambda: self.select_area_and_save(self.canvas_area_var)).grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkLabel(selection_frame, textvariable=self.canvas_area_var).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        row_idx += 1

        image_frame = ctk.CTkFrame(self)
        image_frame.grid(row=row_idx, column=0, padx=10, pady=5, sticky="ew")
        image_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(image_frame, text="2. 그림 업로드", command=lambda: self.upload_image()).grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkLabel(image_frame, textvariable=self.image_path_var, wraplength=320).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        row_idx += 1

        model_layers_frame = ctk.CTkFrame(self)
        model_layers_frame.grid(row=row_idx, column=0, padx=10, pady=10, sticky="ew")
        model_layers_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(model_layers_frame, text="모델 레이어", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=3, pady=(5,0))
        layer_count_frame = ctk.CTkFrame(model_layers_frame, fg_color="transparent")
        layer_count_frame.grid(row=1, column=0, columnspan=3, pady=5, sticky="ew")
        layer_count_frame.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(layer_count_frame, text="레이어 개수:").grid(row=0, column=0, padx=5)
        ctk.CTkEntry(layer_count_frame, textvariable=self.num_layers_var, width=50).grid(row=0, column=1, padx=5)
        ctk.CTkButton(layer_count_frame, text="레이어 업데이트", command=self.update_layer_widgets).grid(row=0, column=2, padx=5, sticky="w")
        self.layer_scroll_frame = ctk.CTkScrollableFrame(model_layers_frame, label_text="Layers")
        self.layer_scroll_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5)
        self.layer_scroll_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(model_layers_frame, text="조합 방식:").grid(row=3, column=0, padx=5, pady=10, sticky="w")
        ctk.CTkOptionMenu(model_layers_frame, variable=self.combination_method_var, values=["Union (Combine)", "Overlay"]).grid(row=3, column=1, columnspan=2, padx=5, pady=10, sticky="ew")
        row_idx += 1

        general_settings_frame = ctk.CTkFrame(self); general_settings_frame.grid(row=row_idx, column=0, padx=10, pady=10, sticky="ew")
        general_settings_frame.grid_columnconfigure(1, weight=1)
        gs_row = 0
        ctk.CTkLabel(general_settings_frame, text="정밀도 설정:").grid(row=gs_row, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkOptionMenu(general_settings_frame, variable=self.precision_var, values=["100%", "90%", "80%", "70%", "60%", "50%", "40%", "30%", "20%", "10%"]).grid(row=gs_row, column=1, columnspan=2, padx=5, pady=5, sticky="ew"); gs_row += 1
        ctk.CTkLabel(general_settings_frame, text="Contour 검색 방법:").grid(row=gs_row, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkOptionMenu(general_settings_frame, variable=self.contour_mode_var, values=["외곽선만 찾기", "모든 선 찾기", "모든 선 찾기 + 계층"]).grid(row=gs_row, column=1, columnspan=2, padx=5, pady=5, sticky="ew"); gs_row += 1
        ctk.CTkLabel(general_settings_frame, text="Contour 근사 방법:").grid(row=gs_row, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkOptionMenu(general_settings_frame, variable=self.contour_method_var, values=["선 압축하기", "모든 점 저장하기"]).grid(row=gs_row, column=1, columnspan=2, padx=5, pady=5, sticky="ew"); gs_row += 1
        ctk.CTkLabel(general_settings_frame, text="Line Epsilon (px):").grid(row=gs_row, column=0, padx=5, pady=2, sticky="w")
        ctk.CTkEntry(general_settings_frame, textvariable=self.line_epsilon_var).grid(row=gs_row, column=1, columnspan=2, padx=5, pady=2, sticky="ew"); gs_row += 1
        ctk.CTkLabel(general_settings_frame, text="Line Delay:").grid(row=gs_row, column=0, padx=5, pady=2, sticky="w")
        ctk.CTkEntry(general_settings_frame, textvariable=self.line_delay_var).grid(row=gs_row, column=1, columnspan=2, padx=5, pady=2, sticky="ew"); gs_row += 1
        ctk.CTkLabel(general_settings_frame, text="마우스 속도:").grid(row=gs_row, column=0, padx=5, pady=5, sticky="w")
        def update_speed_label(value): self.speed_value_label.configure(text=f"{value:.4f}초")
        self.speed_slider = ctk.CTkSlider(general_settings_frame, from_=0.0, to=0.02, variable=self.mouse_duration_var, command=update_speed_label)
        self.speed_slider.grid(row=gs_row, column=1, padx=5, pady=5, sticky="ew")
        self.speed_value_label = ctk.CTkLabel(general_settings_frame, text=""); self.speed_value_label.grid(row=gs_row, column=2, padx=5, pady=5, sticky="w")
        row_idx += 1

        action_frame = ctk.CTkFrame(self); action_frame.grid(row=row_idx, column=0, padx=10, pady=10, sticky="ew")
        action_frame.grid_columnconfigure(0, weight=1); action_frame.grid_columnconfigure(1, weight=1)
        self.start_button = ctk.CTkButton(action_frame, text="3. 미리보기 및 편집", command=self.start_drawing_process); self.start_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.stop_button = ctk.CTkButton(action_frame, text="중지 (ESC 키)", command=self.stop_drawing, state="disabled"); self.stop_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        row_idx += 1

        progress_frame = ctk.CTkFrame(self); progress_frame.grid(row=row_idx, column=0, padx=10, pady=10, sticky="ew")
        progress_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(progress_frame, text="진행률:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ctk.CTkProgressBar(progress_frame, variable=self.total_progress_var).grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        ctk.CTkLabel(progress_frame, textvariable=self.total_status_var).grid(row=0, column=2, padx=5, pady=2, sticky="e")
        row_idx += 1

        log_frame = ctk.CTkFrame(self); log_frame.grid(row=row_idx, column=0, padx=10, pady=10, sticky="nsew")
        self.grid_rowconfigure(row_idx, weight=1)
        self.log_textbox = ctk.CTkTextbox(log_frame, state="disabled", wrap=tk.WORD); self.log_textbox.pack(expand=True, fill="both", padx=5, pady=5)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.load_initial_settings()
        update_speed_label(self.mouse_duration_var.get())

    def update_layer_widgets(self):
        try: num_layers = int(self.num_layers_var.get()); num_layers = max(0, num_layers)
        except ValueError: num_layers = 0; self.num_layers_var.set("0")
        for widget in self.layer_scroll_frame.winfo_children(): widget.destroy()
        old_layers, self.layers = self.layers, []
        model_options = ["HED", "Canny", "Lineart Anime", "SoftEdge"]
        for i in range(num_layers):
            vals = (old_layers[i]["enabled"].get(), old_layers[i]["model"].get(), old_layers[i]["threshold"].get()) if i < len(old_layers) else (True, "HED", "0.5")
            layer = {"enabled": ctk.BooleanVar(value=vals[0]), "model": ctk.StringVar(value=vals[1]), "threshold": ctk.StringVar(value=vals[2])}
            self.layers.append(layer)
            ctk.CTkCheckBox(self.layer_scroll_frame, text=f"L{i+1}", variable=layer["enabled"]).grid(row=i, column=0, padx=5, pady=5)
            ctk.CTkOptionMenu(self.layer_scroll_frame, variable=layer["model"], values=model_options).grid(row=i, column=1, padx=5, pady=5, sticky="ew")
            ctk.CTkEntry(self.layer_scroll_frame, textvariable=layer["threshold"], placeholder_text="Threshold").grid(row=i, column=2, padx=5, pady=5, sticky="ew")
    
    def setup_logging(self):
        self.logger = logging.getLogger(); self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s', '%H:%M:%S')
        if self.logger.hasHandlers(): self.logger.handlers.clear()
        for handler in list(logging.getLogger().handlers):
            if isinstance(handler, logging.StreamHandler): logging.getLogger().removeHandler(handler)
        logging.getLogger().propagate = False
        gui_handler = CtkTextboxHandler(self.log_textbox); gui_handler.setFormatter(formatter); self.logger.addHandler(gui_handler)

    def start_drawing_process(self):
        if self.canvas_area_var.get() == "0,0,0,0" or not self.image_path_var.get():
            self.logger.error("캔버스 영역 선택과 이미지 업로드를 완료해야 합니다."); return
        self.save_current_settings()
        self.start_button.configure(state="disabled", text="처리 중...")
        self.logger.info("이미지 처리 스레드를 시작합니다...")
        processing_thread = threading.Thread(target=self._processing_thread_target, daemon=True); processing_thread.start()

    def _processing_thread_target(self):
        try:
            pipeline = [{"model": l["model"].get(), "threshold": float(l["threshold"].get())} for l in self.layers if l["enabled"].get()]
            if not pipeline:
                self.logger.error("활성화된 모델 레이어가 없습니다."); self.after(0, self.reset_start_button); return

            self.preview_image = fn.generate_preview_image(
                image_path=self.image_path_var.get(), pipeline=pipeline, combination_method=self.combination_method_var.get(),
                canvas_coords=tuple(map(int, self.canvas_area_var.get().split(","))), precision=int(self.precision_var.get().replace("%", "")), 
                logger=self.logger, line_epsilon=float(self.line_epsilon_var.get()),
                contour_mode=self.contour_mode_var.get(), contour_method=self.contour_method_var.get()
            )
            if self.preview_image is None:
                self.logger.error("이미지 처리 실패. 그릴 내용이 없거나 오류 발생."); self.after(0, self.reset_start_button); return
            self.logger.info("이미지 처리 완료. 편집 창을 표시합니다."); self.after(0, self.show_editor_window)
        except Exception as e:
            self.logger.error(f"이미지 처리 중 예외 발생: {e}"); self.after(0, self.reset_start_button)

    def show_editor_window(self):
        editor = fn.EditorWindow(root=self, initial_image=self.preview_image, line_epsilon=float(self.line_epsilon_var.get()),
            contour_mode=self.contour_mode_var.get(), contour_method=self.contour_method_var.get(),
            start_callback=self.finalize_and_start_drawing, cancel_callback=self.cancel_drawing)
        self.wait_window(editor)
        # 편집기 창이 닫힌 후 항상 버튼을 리셋
        self.reset_start_button()

    def finalize_and_start_drawing(self, final_plan):
        self.drawing_plan = final_plan
        if not self.drawing_plan: self.logger.error("편집 후 그릴 선이 없습니다."); return
        self.confirm_start_drawing()

    def confirm_start_drawing(self):
        global drawing_thread
        self.logger.info("자동 그리기 (라인 모드)을 시작합니다."); self.attributes("-topmost", True)
        stop_drawing_flag.clear(); self.stop_button.configure(state="normal")
        drawer = fn.AutoDrawerLine(self.drawing_plan, self.canvas_area_var.get(), self.update_progress, stop_drawing_flag, self.logger,
            float(self.line_delay_var.get()), self.mouse_duration_var.get())
        drawing_thread = threading.Thread(target=drawer.run, daemon=True); drawing_thread.start()
        self.check_thread_status()
    
    def reset_start_button(self):
        self.start_button.configure(state="normal", text="3. 미리보기 및 편집")

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")])
        if path: self.image_path_var.set(path); self.logger.info(f"이미지 선택: {path}")

    def select_area_and_save(self, var):
        self.withdraw(); time.sleep(0.2); area, _ = fn.select_area(self); self.deiconify()
        if area: var.set(f"{area[0]},{area[1]},{area[2]},{area[3]}"); self.logger.info(f"영역 저장: {var.get()}")

    def save_current_settings(self): fn.save_settings(self)
    def load_initial_settings(self):
        settings = fn.load_settings()
        self.canvas_area_var.set(settings.get("CANVAS_AREA", "0,0,0,0"))
        self.image_path_var.set(settings.get("IMAGE_PATH", ""))
        self.precision_var.set(settings.get("PRECISION", "80%"))
        self.line_epsilon_var.set(settings.get("LINE_EPSILON", "1.5"))
        self.line_delay_var.set(settings.get("LINE_DELAY", "0.01"))
        self.mouse_duration_var.set(float(settings.get("MOUSE_DURATION", "0.005")))
        self.contour_mode_var.set(settings.get("CONTOUR_MODE", "외곽선만 찾기"))
        self.contour_method_var.set(settings.get("CONTOUR_METHOD", "선 압축하기"))
        self.combination_method_var.set(settings.get("COMBINATION_METHOD", "Overlay"))
        self.num_layers_var.set(settings.get("NUM_LAYERS", "1"))
        self.update_layer_widgets()
        for i in range(len(self.layers)):
            self.layers[i]["enabled"].set(settings.get(f"L{i+1}_ENABLED", "true" if i==0 else "false").lower() == "true")
            self.layers[i]["model"].set(settings.get(f"L{i+1}_MODEL", "HED"))
            self.layers[i]["threshold"].set(settings.get(f"L{i+1}_THRESHOLD", "0.5"))

    def cancel_drawing(self): self.reset_progress(); self.logger.info("그리기 작업이 취소되었습니다.")
    def stop_drawing(self):
        if drawing_thread and drawing_thread.is_alive():
            self.logger.info("사용자가 그리기를 중지했습니다."); self.attributes("-topmost", False); stop_drawing_flag.set(); self.stop_button.configure(state="disabled")

    def check_thread_status(self):
        if drawing_thread and drawing_thread.is_alive(): self.after(100, self.check_thread_status)
        else: self.attributes("-topmost", False); self.start_button.configure(state="normal"); self.stop_button.configure(state="disabled")

    def update_progress(self, line_status, line_pct, total_status, total_pct):
        self.total_status_var.set(total_status); self.total_progress_var.set(total_pct / 100)
    def reset_progress(self): self.total_status_var.set("대기"); self.total_progress_var.set(0)
    def on_press(self, key):
        if key == keyboard.Key.esc: self.stop_drawing()
    def start_keyboard_listener(self): listener = keyboard.Listener(on_press=self.on_press); listener.start()
    def on_closing(self): self.save_current_settings(); self.logger.info("프로그램을 종료합니다."); self.destroy()

if __name__ == "__main__":
    app = LineDrawApp()
    app.setup_logging()
    app.start_keyboard_listener()
    app.mainloop()