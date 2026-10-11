import sys
import math
import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QComboBox, QSlider, QLabel, 
                               QCheckBox, QGroupBox, QScrollArea, QGridLayout, 
                               QLineEdit, QTabWidget, QFrame)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter

# --- Colormap Utility ---
def make_colormap(colors):
    c = np.array(colors, dtype=np.float32)
    x = np.linspace(0, len(c)-1, 256)
    xp = np.arange(len(c))
    r = np.interp(x, xp, c[:,0])
    g = np.interp(x, xp, c[:,1])
    b = np.interp(x, xp, c[:,2])
    return np.stack([r, g, b], axis=-1).astype(np.uint8)

class GridCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(600, 600)
        self.setStyleSheet("background-color: #11111b; border-radius: 8px;")

    def update_grid(self, rgb_data, smooth):
        rgb_data = np.ascontiguousarray(rgb_data)
        h, w, _ = rgb_data.shape
        q_image = QImage(rgb_data.data, w, h, 3 * w, QImage.Format_RGB888)
            
        transform = Qt.SmoothTransformation if smooth else Qt.FastTransformation
        self.pixmap = QPixmap.fromImage(q_image).scaled(
            self.width(), self.height(), Qt.KeepAspectRatio, transform
        )
        self.update()

    def paintEvent(self, event):
        if not hasattr(self, 'pixmap'): return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        x = (self.width() - self.pixmap.width()) // 2
        y = (self.height() - self.pixmap.height()) // 2
        painter.drawPixmap(x, y, self.pixmap)

class CVCAMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unified CA & CV Engine V9 (Math & Structural Kernels)")
        self.resize(1300, 850)

        self.w2d, self.h2d = 220, 220
        self.w3d, self.h3d, self.d3d = 48, 48, 48 
        
        self.grid_2d = np.zeros((self.h2d, self.w2d), dtype=np.float32)
        self.grid_gs = np.zeros((self.h2d, self.w2d, 2), dtype=np.float32)
        self.grid_rgb = np.zeros((self.h2d, self.w2d, 3), dtype=np.float32)
        self.grid_3d = np.zeros((self.d3d, self.h3d, self.w3d), dtype=np.float32)
        
        self.time_step = 0
        
        self.active_cv_kernel = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32)
        self.active_ca_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.float32)
        self.current_matrix_3d = np.zeros((3,3,3), dtype=np.float32)

        self.colormaps = {
            "Plasma": make_colormap([[13, 8, 135], [126, 3, 168], [203, 70, 121], [248, 149, 64], [240, 249, 33]]),
            "Inferno": make_colormap([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]]),
            "Aurora": make_colormap([[5, 10, 30], [0, 100, 150], [0, 255, 150], [150, 255, 200], [255, 255, 255]]),
            "Magma": make_colormap([[0, 0, 0], [50, 0, 50], [150, 30, 80], [220, 100, 40], [250, 200, 100], [255, 255, 200]]),
            "Hologram": make_colormap([[0, 20, 40], [0, 150, 200], [0, 255, 255], [255, 255, 255]]),
            "Rainbow": make_colormap([[255,0,0], [255,255,0], [0,255,0], [0,255,255], [0,0,255], [255,0,255]]),
            "Cyberpunk": make_colormap([[10, 10, 40], [255, 0, 200], [0, 255, 255], [255, 255, 255]]),
            "Ice": make_colormap([[5, 10, 30], [30, 80, 180], [100, 200, 255], [255, 255, 255]]),
        }

        self.init_ui()
        self.apply_preset("BZ Technicolor Vortex")
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_simulation)
        self.update_canvas()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)

        self.canvas = GridCanvas()
        main_layout.addWidget(self.canvas, stretch=1)

        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(450)
        main_layout.addWidget(self.tabs)

        # --- TAB 1: Engine & Presets ---
        tab1 = QWidget()
        t1_layout = QVBoxLayout(tab1)
        self.tabs.addTab(tab1, "Engine")

        preset_group = QGroupBox("Instant Presets")
        p_layout = QVBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom", "BZ Technicolor Vortex", "3D Lenia Orbium", "Anisotropic Wood Grain", "Gray-Scott Coral", "Digital Maze", "Glitch Art", "Neon Liquid", "Bubbly LoG", "Knight's Glitch", "Honeycomb Hex", "Diamond Lattice", "Pyramid Steps", "Gabor Waves", "Sinc Ripples", "3D Torus Orbium"])
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)
        p_layout.addWidget(self.preset_combo)
        preset_group.setLayout(p_layout)
        t1_layout.addWidget(preset_group)

        sim_group = QGroupBox("Simulation Mode")
        sim_layout = QVBoxLayout()
        sim_layout.addWidget(QLabel("Engine Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["2D Single Channel", "2D Gray-Scott", "2D Multi-Channel (BZ/RPS)", "3D Single Channel (Orbium)"])
        sim_layout.addWidget(self.mode_combo)

        sim_layout.addWidget(QLabel("Activation Function (Single Channel):"))
        self.act_combo = QComboBox()
        self.act_combo.addItems(["Continuous (Reaction-Diffusion)", "Lenia (Gaussian Growth)", "Binary Threshold", "Cyclic (Modulo)"])
        sim_layout.addWidget(self.act_combo)
        sim_group.setLayout(sim_layout)
        t1_layout.addWidget(sim_group)

        dyn_group = QGroupBox("Dynamics & LFOs")
        dyn_layout = QVBoxLayout()
        dyn_layout.addWidget(QLabel("Rule Growth Rate:"))
        self.rule_rate_slider = QSlider(Qt.Horizontal)
        self.rule_rate_slider.setRange(0, 100)
        self.rule_rate_slider.setValue(50)
        dyn_layout.addWidget(self.rule_rate_slider)

        dyn_layout.addWidget(QLabel("Activation Threshold:"))
        thresh_layout = QHBoxLayout()
        self.ca_thresh_slider = QSlider(Qt.Horizontal)
        self.ca_thresh_slider.setRange(1, 99)
        self.ca_thresh_slider.setValue(50)
        thresh_layout.addWidget(self.ca_thresh_slider)
        self.ca_thresh_auto = QCheckBox("Auto (LFO)")
        thresh_layout.addWidget(self.ca_thresh_auto)
        dyn_layout.addLayout(thresh_layout)

        dyn_layout.addWidget(QLabel("Gray-Scott Feed (F):"))
        feed_layout = QHBoxLayout()
        self.feed_slider = QSlider(Qt.Horizontal)
        self.feed_slider.setRange(0, 100)
        feed_layout.addWidget(self.feed_slider)
        self.feed_auto = QCheckBox("Auto (LFO)")
        feed_layout.addWidget(self.feed_auto)
        dyn_layout.addLayout(feed_layout)

        dyn_layout.addWidget(QLabel("Gray-Scott Kill (k):"))
        kill_layout = QHBoxLayout()
        self.kill_slider = QSlider(Qt.Horizontal)
        self.kill_slider.setRange(0, 100)
        kill_layout.addWidget(self.kill_slider)
        dyn_layout.addLayout(kill_layout)
        dyn_group.setLayout(dyn_layout)
        t1_layout.addWidget(dyn_group)
        t1_layout.addStretch()

        # --- TAB 2: Custom Kernel Studio ---
        tab2 = QWidget()
        t2_layout = QVBoxLayout(tab2)
        self.tabs.addTab(tab2, "Kernel Studio")

        kern_group = QGroupBox("Matrix Editor (2D & 3D)")
        k_layout = QVBoxLayout()

        dim_layout = QHBoxLayout()
        dim_layout.addWidget(QLabel("Dimensions:"))
        self.kern_3d_check = QCheckBox("Enable 3D (Z-layers)")
        k_layout.addWidget(self.kern_3d_check)
        k_size_layout = QHBoxLayout()
        k_size_layout.addWidget(QLabel("Size:"))
        self.kern_size_combo = QComboBox()
        self.kern_size_combo.addItems(["3x3", "5x5", "7x7"])
        k_size_layout.addWidget(self.kern_size_combo)
        dim_layout.addLayout(k_size_layout)
        k_layout.addLayout(dim_layout)

        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z-Layer:"))
        self.z_layer_slider = QSlider(Qt.Horizontal)
        self.z_layer_slider.setEnabled(False)
        z_layout.addWidget(self.z_layer_slider)
        self.z_layer_label = QLabel("0")
        z_layout.addWidget(self.z_layer_label)
        k_layout.addLayout(z_layout)

        self.kernel_grid_layout = QGridLayout()
        self.kernel_grid_layout.setSpacing(2)
        k_layout.addLayout(self.kernel_grid_layout)
        self.kernel_edits = []

        aniso_group = QGroupBox("Anisotropic Transform (2D)")
        aniso_layout = QVBoxLayout()
        aniso_layout.addWidget(QLabel("Stretch X:"))
        self.stretch_x_slider = QSlider(Qt.Horizontal)
        self.stretch_x_slider.setRange(10, 100)
        self.stretch_x_slider.setValue(50)
        aniso_layout.addWidget(self.stretch_x_slider)
        aniso_layout.addWidget(QLabel("Stretch Y:"))
        self.stretch_y_slider = QSlider(Qt.Horizontal)
        self.stretch_y_slider.setRange(10, 100)
        self.stretch_y_slider.setValue(50)
        aniso_layout.addWidget(self.stretch_y_slider)
        aniso_layout.addWidget(QLabel("Rotation (Degrees):"))
        self.rot_slider = QSlider(Qt.Horizontal)
        self.rot_slider.setRange(0, 180)
        aniso_layout.addWidget(self.rot_slider)
        aniso_group.setLayout(aniso_layout)
        k_layout.addWidget(aniso_group)

        gen_layout = QHBoxLayout()
        gen_layout.addWidget(QLabel("Generator:"))
        self.gen_combo = QComboBox()
        self.gen_combo.addItems([
            "Gaussian", "Ring", "Laplacian", "Sobel X", "Sobel Y", "Emboss", "Motion Blur", "Plus", "Cross", "Diagonal",
            "Von Neumann", "Moore", "Box Blur", "Sharpen (Strong)", "Prewitt X", "Prewitt Y", "Kirsch N", "LoG (5x5)",
            "Knight's Move", "Hexagonal", "Diagonal Neighbors", "Kirsch E", "Kirsch S", "Kirsch W", "Binomial Blur", "Square Outline", "Unsharp Mask", "High-Pass",
            "Disk (Solid)", "Diamond", "Chebyshev Square", "Cone", "Checkerboard (5x5)", "Roberts Cross", "Scharr X", "Pyramid (5x5)", "Arrow (Right)",
            "Scharr Y", "Laplacian 4-N", "Sobel Diag", "Vertical Line", "Horizontal Line", "Thick Plus", "Circular Blur", "Gabor Filter", "Sinc Wave", "3D Torus"
        ])
        gen_layout.addWidget(self.gen_combo)
        self.gen_btn = QPushButton("Generate")
        self.gen_btn.clicked.connect(self.generate_from_combo)
        gen_layout.addWidget(self.gen_btn)
        k_layout.addLayout(gen_layout)

        self.norm_check = QCheckBox("Normalize Kernel (Sum = 1.0)")
        self.norm_check.setChecked(True)
        k_layout.addWidget(self.norm_check)

        apply_layout = QHBoxLayout()
        self.apply_cv_btn = QPushButton(">>> Set as CV Filter")
        self.apply_cv_btn.clicked.connect(lambda: self.apply_custom_kernel("cv"))
        apply_layout.addWidget(self.apply_cv_btn)
        self.apply_ca_btn = QPushButton(">>> Set as CA Rule")
        self.apply_ca_btn.clicked.connect(lambda: self.apply_custom_kernel("ca"))
        apply_layout.addWidget(self.apply_ca_btn)
        k_layout.addLayout(apply_layout)
        
        kern_group.setLayout(k_layout)
        t2_layout.addWidget(kern_group)

        # --- TAB 3: Visuals & Post-Processing ---
        tab3 = QWidget()
        t3_layout = QVBoxLayout(tab3)
        self.tabs.addTab(tab3, "Visuals")

        vis_group = QGroupBox("Visuals & Playback")
        vis_layout = QVBoxLayout()
        
        vis_layout.addWidget(QLabel("Colormap (Grayscale Modes):"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(self.colormaps.keys())
        vis_layout.addWidget(self.cmap_combo)

        self.smooth_check = QCheckBox("Smooth Glow (Bilinear)")
        self.smooth_check.setChecked(True)
        vis_layout.addWidget(self.smooth_check)

        self.color_cycle_check = QCheckBox("Color Cycling (Animated Hue)")
        vis_layout.addWidget(self.color_cycle_check)
        vis_group.setLayout(vis_layout)
        t3_layout.addWidget(vis_group)

        pp_group = QGroupBox("Post-Processing Pipeline (FX)")
        pp_layout = QVBoxLayout()

        self.bloom_check = QCheckBox("Enable Bloom (Neon Glow)")
        self.bloom_check.setChecked(False)
        pp_layout.addWidget(self.bloom_check)

        pp_layout.addWidget(QLabel("Bloom Intensity:"))
        self.bloom_slider = QSlider(Qt.Horizontal)
        self.bloom_slider.setRange(0, 100)
        self.bloom_slider.setValue(50)
        pp_layout.addWidget(self.bloom_slider)

        self.shade_check = QCheckBox("Enable 3D Topographic Shading")
        self.shade_check.setChecked(False)
        pp_layout.addWidget(self.shade_check)

        pp_layout.addWidget(QLabel("Shade Depth/Z:"))
        self.shade_slider = QSlider(Qt.Horizontal)
        self.shade_slider.setRange(10, 100)
        self.shade_slider.setValue(40)
        pp_layout.addWidget(self.shade_slider)

        self.aberration_check = QCheckBox("Enable Chromatic Aberration (Glitch)")
        self.aberration_check.setChecked(False)
        pp_layout.addWidget(self.aberration_check)

        pp_layout.addWidget(QLabel("Aberration Offset:"))
        self.aberration_slider = QSlider(Qt.Horizontal)
        self.aberration_slider.setRange(1, 20)
        self.aberration_slider.setValue(5)
        pp_layout.addWidget(self.aberration_slider)
        
        pp_group.setLayout(pp_layout)
        t3_layout.addWidget(pp_group)

        sim_ctrl_group = QGroupBox("Playback")
        sc_layout = QVBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setStyleSheet("font-weight: bold; font-size: 16px; padding: 10px;")
        self.play_btn.clicked.connect(self.toggle_play)
        sc_layout.addWidget(self.play_btn)

        row = QHBoxLayout()
        self.random_btn = QPushButton("Randomize")
        self.random_btn.clicked.connect(self.seed_random_blobs)
        row.addWidget(self.random_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_grid)
        row.addWidget(self.clear_btn)
        sc_layout.addLayout(row)

        sc_layout.addWidget(QLabel("Speed (FPS):"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(5, 60)
        self.speed_slider.setValue(20)
        self.speed_slider.valueChanged.connect(self.update_speed)
        sc_layout.addWidget(self.speed_slider)
        sim_ctrl_group.setLayout(sc_layout)
        t3_layout.addWidget(sim_ctrl_group)

        self.kern_3d_check.toggled.connect(self.toggle_3d_kernel)
        self.z_layer_slider.valueChanged.connect(self.update_z_layer_label)
        self.kern_size_combo.currentIndexChanged.connect(self.build_kernel_editor)
        self.z_layer_slider.valueChanged.connect(self.load_z_layer_to_ui)

    def toggle_3d_kernel(self, checked):
        self.z_layer_slider.setEnabled(checked)
        size = int(self.kern_size_combo.currentText()[0])
        if checked:
            self.z_layer_slider.setRange(0, size - 1)
            self.current_matrix_3d = np.zeros((size, size, size), dtype=np.float32)
        self.build_kernel_editor()

    def update_z_layer_label(self, val):
        self.z_layer_label.setText(str(val))

    def build_kernel_editor(self):
        while self.kernel_grid_layout.count():
            item = self.kernel_grid_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.kernel_edits.clear()

        size = int(self.kern_size_combo.currentText()[0])
        for r in range(size):
            for c in range(size):
                edit = QLineEdit("0.0")
                edit.setFixedWidth(45)
                edit.setAlignment(Qt.AlignCenter)
                self.kernel_grid_layout.addWidget(edit, r, c)
                self.kernel_edits.append(edit)
        
        if self.kern_3d_check.isChecked():
            self.z_layer_slider.setRange(0, size - 1)
            self.current_matrix_3d = np.zeros((size, size, size), dtype=np.float32)
            self.z_layer_slider.setValue(size // 2)
        
        self.generate_from_combo()

    def load_z_layer_to_ui(self):
        if not self.kern_3d_check.isChecked(): return
        z = self.z_layer_slider.value()
        size = int(self.kern_size_combo.currentText()[0])
        for r in range(size):
            for c in range(size):
                idx = r * size + c
                self.kernel_edits[idx].setText(f"{self.current_matrix_3d[z, r, c]:.3f}")

    def save_current_layer_to_matrix(self):
        if not self.kern_3d_check.isChecked(): return
        z = self.z_layer_slider.value()
        size = int(self.kern_size_combo.currentText()[0])
        for r in range(size):
            for c in range(size):
                idx = r * size + c
                try:
                    self.current_matrix_3d[z, r, c] = float(self.kernel_edits[idx].text())
                except ValueError:
                    self.current_matrix_3d[z, r, c] = 0.0

    def generate_from_combo(self):
        self.generate_kernel(self.gen_combo.currentText())

    def generate_kernel(self, type_name):
        size = int(self.kern_size_combo.currentText()[0])
        half = size // 2
        
        if self.kern_3d_check.isChecked():
            z, y, x = np.ogrid[-half:half+1, -half:half+1, -half:half+1]
            dist = np.sqrt(x**2 + y**2 + z**2)
            manhattan = np.abs(x) + np.abs(y) + np.abs(z)
            chebyshev = np.maximum(np.maximum(np.abs(x), np.abs(y)), np.abs(z))
            dist_xy = np.sqrt(x**2 + y**2)
            
            if type_name == "Gaussian":
                sigma = half / 2.0
                matrix = np.exp(-(dist**2) / (2 * sigma**2))
            elif type_name == "Ring":
                radius = half / 2.0
                sigma = max(0.5, half / 4.0)
                matrix = np.exp(-((dist - radius)**2) / (2 * sigma**2))
            elif type_name == "3D Torus":
                r_torus = half / 2.0
                sigma = max(0.5, half / 5.0)
                matrix = np.exp(-((dist_xy - r_torus)**2 + z**2) / (2 * sigma**2))
            elif type_name in ["Moore", "Box Blur"]:
                matrix = np.ones((size, size, size))
                if type_name == "Moore": matrix[half, half, half] = 0.0
            elif type_name == "Circular Blur":
                matrix = np.where(dist <= half, 1.0, 0.0)
            elif type_name == "Sinc Wave":
                matrix = np.sinc(dist / 2.0)
            elif type_name == "Von Neumann":
                matrix = np.zeros((size, size, size))
                matrix[half-1, half, half] = 1.0 if size>1 else 0
                matrix[half+1, half, half] = 1.0 if size>1 else 0
                matrix[half, half-1, half] = 1.0 if size>1 else 0
                matrix[half, half+1, half] = 1.0 if size>1 else 0
                matrix[half, half, half-1] = 1.0 if size>1 else 0
                matrix[half, half, half+1] = 1.0 if size>1 else 0
            elif type_name == "Laplacian":
                matrix = np.zeros((size, size, size))
                matrix[half, half, half] = -6.0
                matrix[half-1, half, half] = 1.0; matrix[half+1, half, half] = 1.0
                matrix[half, half-1, half] = 1.0; matrix[half, half+1, half] = 1.0
                matrix[half, half, half-1] = 1.0; matrix[half, half, half+1] = 1.0
            elif type_name == "Disk (Solid)":
                matrix = np.where(dist <= half, 1.0, 0.0) 
            elif type_name == "Diamond":
                matrix = np.where(manhattan <= half, 1.0, 0.0) 
            elif type_name == "Chebyshev Square":
                matrix = np.where(chebyshev <= half, 1.0, 0.0) 
            elif type_name == "Cone":
                max_dist = np.sqrt(3 * (half**2))
                matrix = np.clip(1.0 - (dist / max_dist), 0, 1)
            else:
                matrix = np.zeros((size, size, size))
                matrix[half, half, half] = 1.0 

            self.current_matrix_3d = matrix.astype(np.float32)
            self.load_z_layer_to_ui()
        else:
            y, x = np.ogrid[-half:half+1, -half:half+1]
            theta = np.radians(self.rot_slider.value())
            sx = self.stretch_x_slider.value() / 50.0
            sy = self.stretch_y_slider.value() / 50.0
            xr = (x * np.cos(theta) - y * np.sin(theta)) / sx
            yr = (x * np.sin(theta) + y * np.cos(theta)) / sy
            dist = np.sqrt(xr**2 + yr**2)
            manhattan = np.abs(xr) + np.abs(yr)
            chebyshev = np.maximum(np.abs(xr), np.abs(yr))
            
            matrix = np.zeros((size, size))
            
            k3 = {
                "Sobel X": [[-1,0,1],[-2,0,2],[-1,0,1]],
                "Sobel Y": [[-1,-2,-1],[0,0,0],[1,2,1]],
                "Sobel Diag": [[0,1,2],[-1,0,1],[-2,-1,0]],
                "Emboss": [[-2,-1,0],[-1,1,1],[0,1,2]],
                "Sharpen (Strong)": [[0,-2,0],[-2,9,-2],[0,-2,0]],
                "Prewitt X": [[-1,0,1],[-1,0,1],[-1,0,1]],
                "Prewitt Y": [[-1,-1,-1],[0,0,0],[1,1,1]],
                "Kirsch N": [[5,5,5],[-3,0,-3],[-3,-3,-3]],
                "Kirsch E": [[-3,-3,5],[-3,0,5],[-3,-3,5]],
                "Kirsch S": [[-3,-3,-3],[-3,0,-3],[5,5,5]],
                "Kirsch W": [[5,-3,-3],[5,0,-3],[5,-3,-3]],
                "Hexagonal": [[1,1,0],[1,0,1],[0,1,1]],
                "Diagonal Neighbors": [[1,0,1],[0,0,0],[1,0,1]],
                "Binomial Blur": [[1,2,1],[2,4,2],[1,2,1]],
                "High-Pass": [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],
                "Roberts Cross": [[1,0,0],[0,-1,0],[0,0,0]],
                "Scharr X": [[-3,0,3],[-10,0,10],[-3,0,3]],
                "Scharr Y": [[-3,-10,-3],[0,0,0],[3,10,3]],
                "Laplacian 4-N": [[0,-1,0],[-1,4,-1],[0,-1,0]],
                "Vertical Line": [[0,1,0],[0,1,0],[0,1,0]],
                "Horizontal Line": [[0,0,0],[1,1,1],[0,0,0]],
                "Arrow (Right)": [[0,0,1],[0,1,1],[0,0,1]],
            }
            k5 = {
                "LoG (5x5)": [[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]],
                "Unsharp Mask": [[1,4,6,4,1],[4,16,24,16,4],[6,24,-476,24,6],[4,16,24,16,4],[1,4,6,4,1]],
                "Square Outline": [[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1]],
                "Knight's Move": [[0,1,0,1,0],[1,0,0,0,1],[0,0,1,0,0],[1,0,0,0,1],[0,1,0,1,0]],
                "Checkerboard (5x5)": [[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1]],
                "Pyramid (5x5)": np.outer([1,2,3,2,1], [1,2,3,2,1]),
                "Thick Plus": [[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]],
                "Gabor Filter": np.exp(-(dist**2)/2.0) * np.cos(2 * xr * 1.5),
                "Circular Blur": np.where(dist <= 2, 1.0, 0.0),
                "Sinc Wave": np.sinc(dist / 2.0),
            }

            if type_name == "Gaussian":
                sigma = half / 2.0
                matrix = np.exp(-(dist**2) / (2 * sigma**2))
            elif type_name == "Ring":
                radius = half / 2.0
                sigma = max(0.5, half / 4.0)
                matrix = np.exp(-((dist - radius)**2) / (2 * sigma**2))
            elif type_name == "Laplacian":
                matrix[half, half] = -4.0 if size==3 else -1.0
                matrix += np.where(np.sqrt(x**2+y**2) <= 1, 1.0, 0.0)
            elif type_name == "Disk (Solid)":
                matrix = np.where(dist <= half, 1.0, 0.0)
            elif type_name == "Diamond":
                matrix = np.where(manhattan <= half, 1.0, 0.0)
            elif type_name == "Chebyshev Square":
                matrix = np.where(chebyshev <= half, 1.0, 0.0)
            elif type_name == "Cone":
                max_dist = np.sqrt(2 * (half**2))
                matrix = np.clip(1.0 - (dist / max_dist), 0, 1)
            elif type_name in k3:
                m = np.array(k3[type_name], dtype=np.float32)
                if size == 3: matrix = m
                else:
                    oy, ox = (size - 3) // 2
                    matrix[oy:oy+3, ox:ox+3] = m
            elif type_name in k5:
                m = np.array(k5[type_name], dtype=np.float32)
                if size == 5: matrix = m
                else:
                    oy, ox = (size - 5) // 2
                    matrix[oy:oy+5, ox:ox+5] = m
            elif type_name == "Motion Blur":
                for i in range(size): matrix[i, i] = 1.0
            elif type_name == "Plus":
                matrix[half, :] = 1.0; matrix[:, half] = 1.0
            elif type_name == "Cross":
                for i in range(size): matrix[i, i] = 1.0; matrix[i, size-1-i] = 1.0
            elif type_name == "Diagonal":
                for i in range(size): matrix[i, i] = 1.0
            elif type_name in ["Moore", "Box Blur"]:
                matrix = np.ones((size, size))
                if type_name == "Moore": matrix[half, half] = 0.0
            elif type_name == "Von Neumann":
                matrix[half, half-1] = 1.0 if size>1 else 0
                matrix[half, half+1] = 1.0 if size>1 else 0
                matrix[half-1, half] = 1.0 if size>1 else 0
                matrix[half+1, half] = 1.0 if size>1 else 0

            for r in range(size):
                for c in range(size):
                    idx = r * size + c
                    self.kernel_edits[idx].setText(f"{matrix[r, c]:.3f}")

    def read_kernel_from_ui(self):
        if self.kern_3d_check.isChecked():
            self.save_current_layer_to_matrix()
            matrix = self.current_matrix_3d.copy()
        else:
            size = int(self.kern_size_combo.currentText()[0])
            matrix = np.zeros((size, size), dtype=np.float32)
            for r in range(size):
                for c in range(size):
                    idx = r * size + c
                    try: val = float(self.kernel_edits[idx].text())
                    except ValueError: val = 0.0
                    matrix[r, c] = val
        if self.norm_check.isChecked():
            s = matrix.sum()
            if s != 0: matrix = matrix / s
        return matrix

    def apply_custom_kernel(self, target):
        k = self.read_kernel_from_ui()
        if target == "cv": self.active_cv_kernel = k
        elif target == "ca": self.active_ca_kernel = k

    def on_preset_changed(self):
        name = self.preset_combo.currentText()
        if name != "Custom": self.apply_preset(name)

    def apply_preset(self, name):
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText(name)
        self.preset_combo.blockSignals(False)

        if name == "BZ Technicolor Vortex":
            self.mode_combo.setCurrentText("2D Multi-Channel (BZ/RPS)")
            self.kern_3d_check.setChecked(False)
            self.kern_size_combo.setCurrentText("3x3")
            self.gen_combo.setCurrentText("Gaussian")
            self.generate_kernel("Gaussian")
            self.apply_custom_kernel("ca")
            self.rule_rate_slider.setValue(40)
            self.bloom_check.setChecked(True)
            self.bloom_slider.setValue(30)
        elif name == "3D Lenia Orbium":
            self.mode_combo.setCurrentText("3D Single Channel (Orbium)")
            self.act_combo.setCurrentText("Lenia (Gaussian Growth)")
            self.kern_3d_check.setChecked(True)
            self.kern_size_combo.setCurrentText("5x5")
            self.gen_combo.setCurrentText("Ring")
            self.generate_kernel("Ring")
            self.apply_custom_kernel("ca")
            self.rule_rate_slider.setValue(30)
            self.ca_thresh_slider.setValue(50)
            self.cmap_combo.setCurrentText("Hologram")
            self.shade_check.setChecked(True)
            self.shade_slider.setValue(50)
        elif name == "3D Torus Orbium":
            self.mode_combo.setCurrentText("3D Single Channel (Orbium)")
            self.act_combo.setCurrentText("Lenia (Gaussian Growth)")
            self.kern_3d_check.setChecked(True)
            self.kern_size_combo.setCurrentText("5x5")
            self.gen_combo.setCurrentText("3D Torus")
            self.generate_kernel("3D Torus")
            self.apply_custom_kernel("ca")
            self.rule_rate_slider.setValue(30)
            self.ca_thresh_slider.setValue(50)
            self.cmap_combo.setCurrentText("Inferno")
            self.shade_check.setChecked(True)
            self.shade_slider.setValue(60)
        elif name == "Bubbly LoG":
            self.mode_combo.setCurrentText("2D Single Channel")
            self.act_combo.setCurrentText("Continuous (Reaction-Diffusion)")
            self.kern_3d_check.setChecked(False)
            self.kern_size_combo.setCurrentText("5x5")
            self.gen_combo.setCurrentText("LoG (5x5)")
            self.generate_kernel("LoG (5x5)")
            self.apply_custom_kernel("ca")
            self.rule_rate_slider.setValue(30)
            self.ca_thresh_slider.setValue(20)
            self.cmap_combo.setCurrentText("Magma")
            self.bloom_check.setChecked(True)
            self.shade_check.setChecked(True)
        elif name == "Knight's Glitch":
            self.mode_combo.setCurrentText("2D Single Channel")
            self.act_combo.setCurrentText("Binary Threshold")
            self.kern_3d_check.setChecked(False)
            self.kern_size_combo.setCurrentText("5x5")
            self.gen_combo.setCurrentText("Knight's Move")
            self.generate_kernel("Knight's Move")
            self.apply_custom_kernel("ca")
            self.rule_rate_slider.setValue(50)
            self.ca_thresh_slider.setValue(20)
            self.cmap_combo.setCurrentText("Cyberpunk")
            self.aberration_check.setChecked(True)
        elif name == "Honeycomb Hex":
            self.mode_combo.setCurrentText("2D Single Channel")
            self.act_combo.setCurrentText("Continuous (Reaction-Diffusion)")
            self.kern_3d_check.setChecked(False)
            self.kern_size_combo.setCurrentText("3x3")
            self.gen_combo.setCurrentText("Hexagonal")
            self.generate_kernel("Hexagonal")
            self.apply_custom_kernel("ca")
            self.rule_rate_slider.setValue(40)
            self.ca_thresh_slider.setValue(30)
            self.cmap_combo.setCurrentText("Aurora")
            self.shade_check.setChecked(True)
        elif name == "Diamond Lattice":
            self.mode_combo.setCurrentText("2D Single Channel")
            self.act_combo.setCurrentText("Continuous (Reaction-Diffusion)")
            self.kern_3d_check.setChecked(False)
            self.kern_size_combo.setCurrentText("7x7")
            self.gen_combo.setCurrentText("Diamond")
            self.generate_kernel("Diamond")
            self.apply_custom_kernel("ca")
            self.rule_rate_slider.setValue(30)
            self.ca_thresh_slider.setValue(25)
            self.cmap_combo.setCurrentText("Ice")
            self.shade_check.setChecked(True)
            self.shade_slider.setValue(70)
        elif name == "Pyramid Steps":
            self.mode_combo.setCurrentText("2D Single Channel")
            self.act_combo.setCurrentText("Cyclic (Modulo)")
            self.kern_3d_check.setChecked(False)
            self.kern_size_combo.setCurrentText("5x5")
            self.gen_combo.setCurrentText("Pyramid (5x5)")
            self.generate_kernel("Pyramid (5x5)")
            self.apply_custom_kernel("ca")
            self.rule_rate_slider.setValue(60)
            self.ca_thresh_slider.setValue(50)
            self.cmap_combo.setCurrentText("Plasma")
            self.color_cycle_check.setChecked(True)
        elif name == "Gabor Waves":
            self.mode_combo.setCurrentText("2D Single Channel")
            self.act_combo.setCurrentText("Continuous (Reaction-Diffusion)")
            self.kern_3d_check.setChecked(False)
            self.kern_size_combo.setCurrentText("5x5")
            self.gen_combo.setCurrentText("Gabor Filter")
            self.generate_kernel("Gabor Filter")
            self.apply_custom_kernel("ca")
            self.rule_rate_slider.setValue(40)
            self.ca_thresh_slider.setValue(35)
            self.cmap_combo.setCurrentText("Aurora")
            self.shade_check.setChecked(True)
            self.shade_slider.setValue(30)
        elif name == "Sinc Ripples":
            self.mode_combo.setCurrentText("2D Single Channel")
            self.act_combo.setCurrentText("Continuous (Reaction-Diffusion)")
            self.kern_3d_check.setChecked(False)
            self.kern_size_combo.setCurrentText("5x5")
            self.gen_combo.setCurrentText("Sinc Wave")
            self.generate_kernel("Sinc Wave")
            self.apply_custom_kernel("ca")
            self.rule_rate_slider.setValue(20)
            self.ca_thresh_slider.setValue(10)
            self.cmap_combo.setCurrentText("Ice")
            self.bloom_check.setChecked(True)
            
        self.seed_random_blobs()

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            self.timer.start(int(1000 / self.speed_slider.value()))
            self.play_btn.setText("Pause")

    def update_speed(self, val):
        if self.timer.isActive(): self.timer.start(int(1000 / val))

    def seed_random_blobs(self):
        mode = self.mode_combo.currentText()
        if mode == "3D Single Channel (Orbium)":
            noise = np.random.rand(self.d3d, self.h3d, self.w3d).astype(np.float32)
            self.grid_3d = gaussian_filter(noise, sigma=2.0)
        elif mode == "2D Multi-Channel (BZ/RPS)":
            self.grid_rgb = np.random.rand(self.h2d, self.w2d, 3).astype(np.float32) * 0.5
        elif mode == "2D Gray-Scott":
            self.grid_gs.fill(0.0)
            self.grid_gs[..., 0] = 1.0 
            noise = np.random.rand(self.h2d, self.w2d).astype(np.float32)
            blurred = gaussian_filter(noise, sigma=3.0)
            self.grid_gs[blurred > 0.6, 1] = 1.0 
        else:
            noise = np.random.rand(self.h2d, self.w2d).astype(np.float32)
            self.grid_2d = gaussian_filter(noise, sigma=3.0)
        self.update_canvas()

    def clear_grid(self):
        mode = self.mode_combo.currentText()
        if mode == "3D Single Channel (Orbium)": self.grid_3d.fill(0.0)
        elif mode == "2D Multi-Channel (BZ/RPS)": self.grid_rgb.fill(0.0)
        elif mode == "2D Gray-Scott": self.grid_gs.fill(0.0)
        else: self.grid_2d.fill(0.0)
        self.update_canvas()

    def step_simulation(self):
        self.time_step += 1
        mode = self.mode_combo.currentText()

        if self.ca_thresh_auto.isChecked():
            self.ca_thresh_slider.setValue(int(50 + 30 * math.sin(self.time_step * 0.02)))
        if self.feed_auto.isChecked():
            self.feed_slider.setValue(int(30 + 10 * math.sin(self.time_step * 0.01)))

        if mode == "2D Single Channel":
            neighbor_map = convolve(self.grid_2d, self.active_ca_kernel, mode='wrap')
            activation = self.act_combo.currentText()
            rate = self.rule_rate_slider.value() / 100.0
            threshold = self.ca_thresh_slider.value() / 100.0

            if activation == "Continuous (Reaction-Diffusion)":
                self.grid_2d += (neighbor_map - threshold) * rate * 0.5
            elif activation == "Lenia (Gaussian Growth)":
                growth = 2 * np.exp(-((neighbor_map - threshold)**2) / (0.15**2)) - 1
                self.grid_2d += growth * rate * 0.1
            self.grid_2d = np.clip(self.grid_2d, 0, 1)
            display = self.grid_2d

        elif mode == "2D Gray-Scott":
            lap_u = convolve(self.grid_gs[..., 0], self.active_ca_kernel, mode='wrap')
            lap_v = convolve(self.grid_gs[..., 1], self.active_ca_kernel, mode='wrap')
            F = self.feed_slider.value() / 1000.0
            k = self.kill_slider.value() / 100.0
            Du, Dv = 1.0, 0.5
            reaction = self.grid_gs[..., 0] * (self.grid_gs[..., 1] ** 2)
            self.grid_gs[..., 0] += (Du * lap_u - reaction + F * (1 - self.grid_gs[..., 0]))
            self.grid_gs[..., 1] += (Dv * lap_v + reaction - (F + k) * self.grid_gs[..., 1])
            self.grid_gs = np.clip(self.grid_gs, 0, 1)
            display = self.grid_gs[..., 1]

        elif mode == "2D Multi-Channel (BZ/RPS)":
            k = self.active_ca_kernel
            if k.ndim == 3: k = k[k.shape[0]//2] 
            lap = convolve(self.grid_rgb, k[:, :, None], mode='wrap')
            R, G, B = self.grid_rgb[..., 0], self.grid_rgb[..., 1], self.grid_rgb[..., 2]
            lap_R, lap_G, lap_B = lap[..., 0], lap[..., 1], lap[..., 2]
            rate = self.rule_rate_slider.value() / 50.0
            dR = lap_R + rate * R * (0.8 - R - 0.5 * G)
            dG = lap_G + rate * G * (0.8 - G - 0.5 * B)
            dB = lap_B + rate * B * (0.8 - B - 0.5 * R)
            self.grid_rgb[..., 0] += dR
            self.grid_rgb[..., 1] += dG
            self.grid_rgb[..., 2] += dB
            self.grid_rgb = np.clip(self.grid_rgb, 0, 1)
            display = self.grid_rgb

        elif mode == "3D Single Channel (Orbium)":
            neighbor_map = convolve(self.grid_3d, self.active_ca_kernel, mode='wrap')
            activation = self.act_combo.currentText()
            rate = self.rule_rate_slider.value() / 100.0
            threshold = self.ca_thresh_slider.value() / 100.0
            if activation == "Lenia (Gaussian Growth)":
                growth = 2 * np.exp(-((neighbor_map - threshold)**2) / (0.15**2)) - 1
                self.grid_3d += growth * rate * 0.1
            else:
                self.grid_3d += (neighbor_map - threshold) * rate * 0.5
            self.grid_3d = np.clip(self.grid_3d, 0, 1)
            display = np.max(self.grid_3d, axis=0)

        self.update_canvas(display)

    def apply_post_processing(self, rgb_data):
        if self.aberration_check.isChecked():
            offset = self.aberration_slider.value()
            rgb_data[..., 0] = np.roll(rgb_data[..., 0], offset, axis=1)
            rgb_data[..., 2] = np.roll(rgb_data[..., 2], -offset, axis=1)

        if self.shade_check.isChecked():
            lum = np.mean(rgb_data, axis=2)
            dy, dx = np.gradient(lum)
            depth = self.shade_slider.value() / 100.0
            shade = 1.0 - (dx * depth) - (dy * depth)
            rgb_data = rgb_data * shade[..., None]
            rgb_data = np.clip(rgb_data, 0, 1)

        if self.bloom_check.isChecked():
            intensity = self.bloom_slider.value() / 100.0
            bright = np.clip(rgb_data - 0.6, 0, 1) * 2.5
            blurred_bright = gaussian_filter(bright, sigma=3.0)
            rgb_data = np.clip(rgb_data + blurred_bright * intensity, 0, 1)

        return rgb_data

    def update_canvas(self, grid=None):
        if grid is None: grid = self.grid_2d
        
        if grid.ndim == 3 and grid.shape[2] == 3:
            rgb_data = grid
        else:
            self.color_shift = (getattr(self, 'color_shift', 0) + (2 if self.color_cycle_check.isChecked() else 0)) % 256
            img_data = (np.clip(grid, 0, 1) * 255).astype(np.int16)
            if self.color_cycle_check.isChecked():
                img_data = (img_data + self.color_shift) % 256
            img_data = img_data.astype(np.uint8)
            cmap = self.colormaps[self.cmap_combo.currentText()]
            rgb_data = cmap[img_data].astype(np.float32) / 255.0

        rgb_data = self.apply_post_processing(rgb_data.copy())
        self.canvas.update_grid(rgb_data, self.smooth_check.isChecked())

    def paint_at(self, pos):
        mode = self.mode_combo.currentText()
        if mode == "3D Single Channel (Orbium)": return 
        
        if 0 <= pos.x() < self.canvas.width() and 0 <= pos.y() < self.canvas.height():
            grid_x = int(pos.x() / self.canvas.width() * self.w2d)
            grid_y = int(pos.y() / self.canvas.height() * self.h2d)
            r = 6
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    if dx**2 + dy**2 <= r**2:
                        ny, nx = grid_y + dy, grid_x + dx
                        if 0 <= ny < self.h2d and 0 <= nx < self.w2d:
                            if mode == "2D Multi-Channel (BZ/RPS)":
                                self.grid_rgb[ny, nx, 0] = 1.0
                            else:
                                self.grid_2d[ny, nx] = 1.0
            self.update_canvas()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: self.paint_at(self.canvas.mapFrom(self, event.pos()))
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton: self.paint_at(self.canvas.mapFrom(self, event.pos()))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QMainWindow { background-color: #1e1e2e; }
        QWidget { color: #cdd6f4; font-family: 'Segoe UI', 'Arial', sans-serif; font-size: 13px; }
        QScrollArea, QScrollArea > QWidget > Widget { background-color: #1e1e2e; }
        QGroupBox { border: 1px solid #45475a; border-radius: 6px; margin-top: 14px; padding-top: 14px; font-weight: bold; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #89b4fa; }
        QPushButton { background-color: #313244; border: 1px solid #45475a; border-radius: 4px; padding: 6px; }
        QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
        QPushButton:pressed { background-color: #11111b; }
        QComboBox { background-color: #313244; border: 1px solid #45475a; border-radius: 4px; padding: 4px; }
        QComboBox:hover { border-color: #89b4fa; }
        QLineEdit { background-color: #313244; border: 1px solid #45475a; border-radius: 4px; padding: 2px; }
        QCheckBox::indicator { border: 1px solid #45475a; background: #313244; width: 14px; height: 14px; border-radius: 3px; }
        QCheckBox::indicator:checked { background: #89b4fa; border-color: #89b4fa; }
        QSlider::groove:horizontal { background: #313244; height: 6px; border-radius: 3px; }
        QSlider::handle:horizontal { background: #89b4fa; width: 14px; margin: -4px 0; border-radius: 7px; }
        QSlider::sub-page:horizontal { background: #89b4fa; border-radius: 3px; }
        QTabWidget::pane { border: 1px solid #45475a; border-radius: 6px; }
        QTabBar::tab { background: #313244; border: 1px solid #45475a; padding: 8px 12px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
        QTabBar::tab:selected { background: #89b4fa; color: #11111b; font-weight: bold; }
    """)
    window = CVCAMainWindow()
    window.show()
    sys.exit(app.exec())