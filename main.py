from time import sleep
from curiosity import curiosity
from camera import camera
import time
from dmx_utils import create_dmx_controller

dmx = create_dmx_controller(port='/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0', device_type='ftdi')

#DMX vals
X=170
Y=0
Xchan=1
Ychan=3

#Rotation limits
X_MIN = 0
X_MAX = 80
Y_MIN = 0
Y_MAX = 80

dmx.update_channel(6, 255)  # Set general DIMMER to maximum
dmx.update_channel(Xchan, X)
dmx.update_channel(10, 12)  # turn the light on
interval = 0.1
dmx.run(interval)

sleep(5)
dmx.update_channel(5, 240)  # set move speed to slow (the greater v the slower)
dmx.run(interval)

#camera and curiosity vals
cameraindex = 0
CAM = camera(cameraindex=cameraindex, preview=True, display_backend="framebuffer")
CAM.start_cam()
sleep(5)
CST = curiosity(CAM, pause=0, split_values=[1,1], visualization_mode="activation_heatmap", movement_grid=[2,2])

def on_reset():
    global X, Y
    X = (X_MIN + X_MAX) / 2
    Y = (Y_MIN + Y_MAX) / 2
    dmx.update_channel(Xchan, int(X))
    dmx.update_channel(Ychan, int(Y))
    dmx.run(interval)

CST.on_reset_start = on_reset
CST.start()

running = True
last_status_print = 0

while running:
    sv = CST.state_vals[:]

    left   = sv[0] + sv[2]
    right  = sv[1] + sv[3]
    top    = sv[0] + sv[1]
    bottom = sv[2] + sv[3]

    diffX = left - right
    diffY = top - bottom

    X += diffX / 4
    Y += diffY / 4
    X = max(X_MIN, min(X_MAX, X))
    Y = max(Y_MIN, min(Y_MAX, Y))

    dmx.update_channel(Xchan, int(X))
    dmx.update_channel(Ychan, int(Y))
    dmx.run(interval)

    now = time.time()
    if now - last_status_print >= 2.0:
        scores = " ".join(f"{v:.1f}" for v in sv)
        print(f"curiosity={scores}  X={int(X)} Y={int(Y)}")
        last_status_print = now

    sleep(0.05)
dmx.close()
