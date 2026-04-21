from time import sleep
import math
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

#Exploration
EXPLORATION_WEIGHT = 10.0  # boost for undervisited regions; raise to explore more
LIMIT_MARGIN = 10          # DMX units from limit where repulsion kicks in
LIMIT_PUSH = 3             # DMX units pushed away from limit per frame

dmx.update_channel(6, 255)  # Set general DIMMER to maximum

dmx.update_channel(Xchan,X) 
dmx.update_channel(10,15) # turn the light on 
#dmx.update_channel(Ychan,Y) 
interval=0.1
dmx.run(interval)

sleep(5)
dmx.update_channel(5, 240) # set move speed to slow (the greater v the slower)
dmx.run(interval)

#camera and curiosity vals
cameraindex=0
#CAM=camera(cameraindex=cameraindex,preview=True,display_backend="window")
CAM = camera(cameraindex=cameraindex, preview=True, display_backend="framebuffer")
CAM.start_cam()
sleep(5)
CST=curiosity(CAM,pause=0,split_values =[1,1], visualization_mode="activation_heatmap", movement_grid=[2,2])
CST.start()

#CSTg=curiosity(CAM,pause=0,split_values =[1,1])
#CSTg.start()


running = True
last_status_print = 0

# UCB1 exploration state: one count per movement grid region (topleft, topright, bottomleft, bottomright)
region_counts = [1] * 4
total_steps = 4

while running:
    # Snapshot so the curiosity thread can't mutate the list mid-loop
    sv = CST.state_vals[:]

    # UCB1 exploration bonus: regions visited less get a higher bonus
    log_total = math.log(total_steps)
    adj = [sv[i] + EXPLORATION_WEIGHT * math.sqrt(log_total / region_counts[i]) for i in range(4)]

    left  = adj[0] + adj[2]
    right = adj[1] + adj[3]
    top   = adj[0] + adj[1]
    bottom= adj[2] + adj[3]

    diffX = left - right
    diffY = top - bottom

    X += diffX / 4
    Y += diffY / 4
    X = max(X_MIN, min(X_MAX, X))
    Y = max(Y_MIN, min(Y_MAX, Y))

    # Repel from limits so the head doesn't lock on boundaries
    if X <= X_MIN + LIMIT_MARGIN:
        X += LIMIT_PUSH
    elif X >= X_MAX - LIMIT_MARGIN:
        X -= LIMIT_PUSH
    if Y <= Y_MIN + LIMIT_MARGIN:
        Y += LIMIT_PUSH
    elif Y >= Y_MAX - LIMIT_MARGIN:
        Y -= LIMIT_PUSH

    # Record which region curiosity pulled toward (raw score, not adjusted)
    winner = sv.index(max(sv))
    region_counts[winner] += 1
    total_steps += 1

    dmx.update_channel(Xchan, int(X))
    dmx.update_channel(Ychan, int(Y))
    dmx.run(interval)

    now = time.time()
    if now - last_status_print >= 2.0:
        scores = " ".join(f"{v:.1f}" for v in sv)
        visits = " ".join(str(c) for c in region_counts)
        print(f"curiosity={scores}  visits={visits}  X={int(X)} Y={int(Y)}")
        last_status_print = now

    sleep(0.05)
dmx.close()

