import os
# Must be set before cv2 is imported to suppress V4L2/backend noise on Linux
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
import cv2
import threading
from queue import Queue
import time
import pygame


class camera:

    # display_backend:
    #   "kmsdrm" -- SDL2 fullscreen via KMS/DRM, no X11 needed (Pi headless)
    #   "fbcon"  -- SDL2 fullscreen via framebuffer console (older Pi / fallback)
    #   "window" -- SDL2 windowed, requires X11 or Wayland (desktop / debug)
    def __init__(self, cameraindex=0, preview=False, cpu_affinity=None, display_backend="kmsdrm"):
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
        self._screen = None
        self._disp_w = self.WIDTH
        self._disp_h = self.HEIGHT

        if self.preview:
            self._init_display()

    def _init_display(self):
        if self.display_backend == "kmsdrm":
            os.environ['SDL_VIDEODRIVER'] = 'kmsdrm'
        elif self.display_backend == "fbcon":
            os.environ['SDL_VIDEODRIVER'] = 'fbcon'
        # "window" leaves SDL_VIDEODRIVER unset so SDL auto-detects X11/Wayland

        try:
            pygame.init()
            if self.display_backend in ("kmsdrm", "fbcon"):
                info = pygame.display.Info()
                w = info.current_w if info.current_w > 0 else self.WIDTH
                h = info.current_h if info.current_h > 0 else self.HEIGHT
                flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
            else:
                w, h = self.WIDTH, self.HEIGHT
                flags = 0

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
        if self._screen is None:
            return
        try:
            resized = cv2.resize(frame, (self._disp_w, self._disp_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            # frombuffer avoids an extra numpy copy; rgb is contiguous after cvtColor
            surf = pygame.image.frombuffer(rgb, (self._disp_w, self._disp_h), 'RGB')
            self._screen.blit(surf, (0, 0))
            pygame.display.flip()
            # Keep SDL event queue drained so the OS doesn't mark the app as hung
            pygame.event.pump()
        except Exception as e:
            print(f"Display write error: {e}")

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
        # CAP_V4L2 targets Linux V4L2 directly, avoiding depth-camera and other
        # backend probes that flood stderr on Pi
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
