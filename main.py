from time import sleep
import statistics as stats
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

dmx.update_channel(6, 255)  # Set general DIMMER to maximum

dmx.update_channel(Xchan,X) 
dmx.update_channel(10,50) # turn the light on 
#dmx.update_channel(Ychan,Y) 
interval=0.1
dmx.run(interval)

sleep(5)
dmx.update_channel(5, 203) # set move speed to slow (the greater v the slower)
dmx.run(interval)

#camera and curiosity vals
cameraindex=0
CAM=camera(cameraindex=cameraindex,preview=True)
CAM.start_cam()
sleep(5)
CST=curiosity(CAM,pause=0,split_values =[1,1], visualization_mode="activation_heatmap", movement_grid=[2,2])
CST.start()

#CSTg=curiosity(CAM,pause=0,split_values =[1,1])
#CSTg.start()


Xhist=[]
Xhist_max=20
meanX=0

movespeed=0.5
boring_target=20
lastX=""
lastGV=0
running=True

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

while running:
    # Snapshot the list once so all reads in this iteration see the same values
    sv = CST.state_vals[:]

    sides={0:"topleft",1:"topright",2:"bottomleft",3:"bottomright"}
    maxv=max(sv)
    maxindex=sv.index(maxv)
    median=sum(sv)/4
    maxside=sides[maxindex]
    ratio=maxv-median
    suma=sum(sv)

    left=sum([sv[0],sv[2]])
    right=sum([sv[1],sv[3]])

    top=sum([sv[0],sv[1]])
    bottom=sum([sv[2],sv[3]])

    diffX=(left-right)
    diffY=(top-bottom)
    #print("difference",diff)

    div=4

    X+=(diffX/div)
    Y+=(diffY/div)
    #print("diff",diffX,diffY)
    #print("X,Y:",X,Y)
    #force max min values
    if X<0:
        X=10
    if X>255:
        X=205
    if Y<0:
       Y=10
    if Y>170:
        Y=150

    dmx.update_channel(Xchan, int(X))
    dmx.update_channel(Ychan, int(Y))
    dmx.run(interval)
    #sleep(interval)
dmx.close()

