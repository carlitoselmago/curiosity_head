import os
# Must be set before cv2 is imported to suppress V4L2/backend noise on Linux
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
import cv2
import threading
from queue import Queue
import numpy as np
import time


class camera:

    # display_backend:
    #   "framebuffer" -- direct /dev/fb0 write, fastest, no X11 or pygame needed
    #   "kmsdrm"      -- pygame/SDL2 fullscreen via KMS/DRM, no X11 needed
    #   "fbcon"       -- pygame/SDL2 fullscreen via framebuffer console
    #   "window"      -- pygame/SDL2 windowed, requires X11 or Wayland
    def __init__(self, cameraindex=0, preview=False, cpu_affinity=None, display_backend="framebuffer"):
        self.preview = preview
        self.display_backend = display_backend
        self.WIDTH = 720
        self.HEIGHT = 576
        self.cameraindex = cameraindex
        self.STREAM_RESOLUTION = str(self.WIDTH) + "x" + str(self.HEIGHT)
        self.frame = None
        self.display_frame = None
        self.curiosity_data = []
        # Default: leave cores 0-1 free for the NN/curiosity thread
        self.cpu_affinity = cpu_affinity if cpu_affinity is not None else {2, 3}
        # Framebuffer state
        self._fb = None
        self._fb_stride = 0
        # Pygame state (lazy-initialised in get_frames thread)
        self._screen = None
        self._disp_w = self.WIDTH
        self._disp_h = self.HEIGHT

    # ------------------------------------------------------------------
    # Direct framebuffer path (/dev/fb0, RGB565)
    # ------------------------------------------------------------------

    def _init_fb(self):
        try:
            with open('/sys/class/graphics/fb0/stride', 'r') as f:
                stride = int(f.read().strip())
            fb_size = stride * self.HEIGHT
            self._fb = np.memmap('/dev/fb0', dtype='uint8', mode='r+', shape=(fb_size,))
            self._fb_stride = stride
            print(f"Framebuffer: {self.WIDTH}x{self.HEIGHT} RGB565 stride={stride} -> /dev/fb0")
        except Exception as e:
            print(f"Framebuffer init failed: {e}")
            self.preview = False
            self._fb = None

    @staticmethod
    def _bgr_to_rgb565(bgr):
        r = bgr[:, :, 2].astype(np.uint16)
        g = bgr[:, :, 1].astype(np.uint16)
        b = bgr[:, :, 0].astype(np.uint16)
        return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)

    def _write_fb(self, frame):
        if self._fb is None:
            return
        try:
            # Camera and framebuffer are both 720x576 -- no resize needed
            rgb565 = self._bgr_to_rgb565(frame)   # (H, W) uint16
            self._fb[:] = rgb565.flatten().view(np.uint8)
        except Exception as e:
            print(f"Framebuffer write error: {e}")

    # ------------------------------------------------------------------
    # Pygame path (kmsdrm / fbcon / window)
    # ------------------------------------------------------------------

    def _init_display(self):
        import pygame

        if self.display_backend == "kmsdrm":
            os.environ['SDL_VIDEODRIVER'] = 'kmsdrm'
        elif self.display_backend == "fbcon":
            os.environ['SDL_VIDEODRIVER'] = 'fbcon'
        # "window" leaves SDL_VIDEODRIVER unset so SDL auto-detects X11/Wayland

        # SDL2 kmsdrm needs XDG_RUNTIME_DIR; systemd services don't set it
        if 'XDG_RUNTIME_DIR' not in os.environ or not os.environ['XDG_RUNTIME_DIR']:
            runtime_dir = f'/run/user/{os.getuid()}'
            if not os.path.exists(runtime_dir):
                runtime_dir = '/tmp'
            os.environ['XDG_RUNTIME_DIR'] = runtime_dir
            print(f"Set XDG_RUNTIME_DIR={runtime_dir}")

        try:
            pygame.init()
            info = pygame.display.Info()
            w = info.current_w if info.current_w > 0 else self.WIDTH
            h = info.current_h if info.current_h > 0 else self.HEIGHT
            if self.display_backend in ("kmsdrm", "fbcon"):
                flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
            else:
                flags = pygame.FULLSCREEN
            self._screen = pygame.display.set_mode((w, h), flags)
            self._disp_w = w
            self._disp_h = h
            pygame.display.set_caption("curiosity")
            pygame.mouse.set_visible(False)
            print(f"Display: {w}x{h} via {self.display_backend} (pygame/SDL2)")
        except Exception as e:
            print(f"Display init failed ({self.display_backend}): {e}")
            self.preview = False
            self._screen = None

    def _write_display(self, frame):
        import pygame
        if self._screen is None:
            return
        try:
            resized = cv2.resize(frame, (self._disp_w, self._disp_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            surf = pygame.image.frombuffer(rgb, (self._disp_w, self._disp_h), 'RGB')
            self._screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.event.pump()
        except Exception as e:
            print(f"Display write error: {e}")

    # ------------------------------------------------------------------
    # Camera capture
    # ------------------------------------------------------------------

    def _scan_camera_index(self):
        """Try V4L2 indices 0-9 and return the first that delivers a frame."""
        for idx in range(10):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    print(f"Auto-detected camera at index {idx}")
                    return idx
            cap.release()
        print("No working camera found in indices 0-9, keeping configured index.")
        return self.cameraindex

    def _open_camera(self):
        cap = cv2.VideoCapture(self.cameraindex, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 15)
        return cap

    def get_frames(self):
        try:
            os.sched_setaffinity(0, self.cpu_affinity)
        except (AttributeError, OSError) as e:
            print(f"Could not set camera thread affinity: {e}")

        if self.preview:
            if self.display_backend == "framebuffer":
                self._init_fb()
            else:
                self._init_display()

        cap = self._open_camera()

        while True:
            if not cap.isOpened():
                print(f"Camera index {self.cameraindex} not available, retrying in 2s...")
                cap.release()
                time.sleep(2)
                cap = self._open_camera()
                continue

            ret, frame = cap.read()

            if not ret:
                print(f"Camera read failed on index {self.cameraindex}, reconnecting...")
                cap.release()
                time.sleep(1)
                cap = self._open_camera()
                continue

            self.frame = frame

            if self.preview:
                blurred = cv2.GaussianBlur(frame, (31, 31), 0)
                preview_frame = self.display_frame if self.display_frame is not None else blurred
                if self.display_backend == "framebuffer":
                    self._write_fb(preview_frame)
                else:
                    self._write_display(preview_frame)
                self.put_with_drop(blurred)
            else:
                self.put_with_drop(frame)

    def start_cam(self):
        self.cameraindex = self._scan_camera_index()
        self.frame_queue = Queue(maxsize=300)
        capture_thread = threading.Thread(target=self.get_frames, daemon=True)
        capture_thread.start()

    def put_with_drop(self, item):
        if self.frame_queue.full():
            self.frame_queue.get()
        self.frame_queue.put(item)
