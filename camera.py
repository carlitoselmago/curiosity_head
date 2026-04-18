import os
# Must be set before cv2 is imported to suppress V4L2/backend noise on Linux
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
import cv2
import threading
from queue import Queue
import numpy as np
import time


class camera:

    def __init__(self, cameraindex=0, preview=False, cpu_affinity=None):
        self.preview = preview
        self.WIDTH = 720
        self.HEIGHT = 576
        self.cameraindex = cameraindex
        self.STREAM_RESOLUTION = str(self.WIDTH) + "x" + str(self.HEIGHT)
        self.frame = None
        self.display_frame = None
        self.curiosity_data = []
        # Default: leave cores 0-1 free for the NN/curiosity thread
        self.cpu_affinity = cpu_affinity if cpu_affinity is not None else {2, 3}
        self._fb = None
        self._fb_width = 0
        self._fb_height = 0
        self._fb_bpp = 0

        if self.preview:
            self._init_framebuffer()

    def _init_framebuffer(self):
        try:
            # Use the physical display mode (e.g. "U:1920x1080-60") rather than
            # virtual_size, which includes extra rows for double-buffering and
            # can report padded widths that differ from the visible area.
            with open('/sys/class/graphics/fb0/modes', 'r') as f:
                mode_line = f.read().strip().split('\n')[0]  # take first mode
            # format: "U:WxH-R" or "U:WxH"
            dims = mode_line.split(':')[-1].split('-')[0]
            w, h = map(int, dims.split('x'))

            with open('/sys/class/graphics/fb0/bits_per_pixel', 'r') as f:
                bpp = int(f.read().strip())

            # stride (bytes per row) may differ from w * bytes_per_pixel due to
            # hardware alignment; use it to size the memmap correctly
            stride_path = '/sys/class/graphics/fb0/stride'
            if os.path.exists(stride_path):
                with open(stride_path, 'r') as f:
                    stride = int(f.read().strip())
            else:
                stride = w * (bpp // 8)

            fb_size = stride * h
            self._fb = np.memmap('/dev/fb0', dtype='uint8', mode='r+', shape=(fb_size,))
            self._fb_width = w
            self._fb_height = h
            self._fb_bpp = bpp
            self._fb_stride = stride
            print(f"Framebuffer: {w}x{h} {bpp}bpp stride={stride} -> /dev/fb0")
        except Exception as e:
            print(f"Framebuffer unavailable: {e}")
            print("  check: ls -la /dev/fb0   cat /sys/class/graphics/fb0/modes")
            print("  if X11/desktop is running it will hide framebuffer output -- boot to console instead")
            self.preview = False
            self._fb = None

    @staticmethod
    def _bgr_to_rgb565(bgr):
        r = bgr[:, :, 2].astype(np.uint16)
        g = bgr[:, :, 1].astype(np.uint16)
        b = bgr[:, :, 0].astype(np.uint16)
        return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)

    def _write_framebuffer(self, frame):
        if self._fb is None:
            return
        resized = cv2.resize(frame, (self._fb_width, self._fb_height))
        try:
            bpp_bytes = self._fb_bpp // 8
            row_bytes = self._fb_width * bpp_bytes

            if self._fb_bpp == 32:
                pixel_data = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)
            elif self._fb_bpp == 16:
                pixel_data = self._bgr_to_rgb565(resized).view(np.uint8).reshape(self._fb_height, row_bytes)
            else:
                print(f"Unsupported framebuffer depth: {self._fb_bpp}bpp, disabling preview.")
                self.preview = False
                self._fb = None
                return

            if self._fb_stride == row_bytes:
                # No padding -- write in one shot
                self._fb[:] = pixel_data.flatten()
            else:
                # Stride differs from row width -- write row by row
                for y in range(self._fb_height):
                    offset = y * self._fb_stride
                    self._fb[offset:offset + row_bytes] = pixel_data[y].flatten()
        except Exception as e:
            print(f"Framebuffer write error: {e}")

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
                self._write_framebuffer(preview_frame)
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
