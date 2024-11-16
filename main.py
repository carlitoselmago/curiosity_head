from time import sleep
import statistics as stats
from curiosity import curiosity
from camera import camera
import time 
from pyDMXController import pyDMXController
dmx = pyDMXController(port='/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0', device_type='ftdi')


#DMX vals
X=170
Y=0
Xchan=1
Ychan=3

dmx.update_channel(6, 255)  # Set general DIMMER to maximum

dmx.update_channel(Xchan,X) 
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
CST=curiosity(CAM,pause=0,split_values =[2,2])
CST.start()

CSTg=curiosity(CAM,pause=0,split_values =[1,1])
CSTg.start()


Xhist=[]
Xhist_max=10
meanX=0

movespeed=0.5
boring_target=20

running=True

while running:
    
    sides={0:"topleft",1:"topright",2:"bottomleft",3:"bottomright"}
    #maxindex=index_min = max(range(len(CST.state_vals)), key=CST.state_vals.__getitem__)
    maxv=max(CST.state_vals)
    maxindex=CST.state_vals.index(maxv)
    median=sum(CST.state_vals)/4
    maxside=sides[maxindex]
    ratio=maxv-median
    suma=sum(CST.state_vals)
    #print(maxside,ratio,CST.state_vals)
    #print()
    general_view=CSTg.state_vals[0]
    print("general_view",general_view)
    
    print("split view",CST.state_vals)
    #print(suma) # overall curiosity
    #print(left,right)


    left=sum([CST.state_vals[0],CST.state_vals[2]])
    right=sum([CST.state_vals[1],CST.state_vals[3]])

    top=sum([CST.state_vals[0],CST.state_vals[1]])
    bottom=sum([CST.state_vals[2],CST.state_vals[3]])

    diffX=(left-right)
    diffY=(top-bottom)
    #print("difference",diff)

    Xhist.append(diffX)
    if len(Xhist)>Xhist_max:
        Xhist.pop(0)
        meanX=stats.mean(Xhist)
        #print("meanX",meanX)

        #NEW APROACH, with separate blocks and single view
        if general_view<3:
            #boriiing
            if meanX>0:
            ##if left>right:
                #print("<-",diff,X,suma)
                #move left
                X+=abs(meanX/4)
            else:
                #print("->",diff,X,suma)
                #move right
                X-=abs(meanX/4)
        else:
            #excited!
            pass

    """
    if ((left+right)<(boring_target*2)):
        #stage boringness complete. lets explore
        if left>right:
            X+=movespeed
        else:
            X-=movespeed
    else:
        # stay and exploit
        #unless the difference is very strong
        print("X diff",diffX)
        if (abs(diffX)>5):
            if diffX>0:
                X+=movespeed
            else:
                X-=movespeed
        pass
    """
    """
    #if (diff>10):
    if meanX>0:
    ##if left>right:
        #print("<-",diff,X,suma)
        #move left
        X+=abs(meanX/4)
    else:
        #print("->",diff,X,suma)
        #move right
        X-=abs(meanX/4)
    
    if top>bottom:
        Y+=movespeed
    else:
        Y-=movespeed
    """
    #force max min values
    if X<0:
        X=1
    if X>255:
        X=255
    if Y<0:
       Y=1
    if Y>170:
        Y=170

    dmx.update_channel(Xchan, int(X))
    dmx.update_channel(Ychan, int(Y))
    dmx.run(interval)
    #sleep(interval)
dmx.close()



"""
for i in range(256):
    X+=1
    dmx.update_channel(Xchan, X)
    dmx.run(interval)

print("now sleep")
time.sleep(5)
"""
