from time import sleep
import statistics as stats
from curiosity import curiosity
from camera import camera
import time 
import numpy as np
from pyDMXController import pyDMXController
dmx = pyDMXController(port='/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0', device_type='ftdi')


#DMX vals
X=170
Y=0
Xchan=1
Ychan=3

dmx.update_channel(6, 255)  # Set general DIMMER to maximum

dmx.update_channel(Xchan,X) 
#dmx.update_channel(10,50) # turn the light on 
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

# general view
CSTg=curiosity(CAM,pause=0,split_values =[1,1])
CSTg.start()


pos={}
posdiv=40 #will be divided between 255 values and converted to ints. The smaller the more detailed
last_pos_v={} #last position value

movespeed=0.5
boring_target=20
lastX=""
lastGV=0
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
    total=CSTg.state_vals[0]
    #print("split view",CST.state_vals)

    left=sum([CST.state_vals[0],CST.state_vals[2]])
    right=sum([CST.state_vals[1],CST.state_vals[3]])

    top=sum([CST.state_vals[0],CST.state_vals[1]])
    bottom=sum([CST.state_vals[2],CST.state_vals[3]])

    diffX=(left-right)
    diffY=(top-bottom)
    #print("difference",diff)

    div=4

    # create a ring of states
    ring=["","","",
          "","","",
           "","",""]
    ringV={}
    c=0
    for x in range(3):
        for y in range(3):
            tX=int(X/posdiv)+(x-1)
            tY=int(Y/posdiv)+(y-1)
            if tX>=0 and tY>=0:
                ring[c]=str(tX)+"_"+str(tY)
                if ring[c] in pos:
                    ringV[ring[c]]=pos[ring[c]]
                else:
                    ringV[ring[c]]=0
            c+=1
    #print("ring",ring)
    # Convert dictionary values to a NumPy array for normalization
    values = np.array(list(ringV.values()))

    # Compute the normalization factor
    norm = np.sqrt(np.sum(values**2))

    # Normalize the values
    normalized_values = values / norm

    # Update the dictionary with normalized values
    ringV = {key: value for key, value in zip(ringV.keys(), normalized_values)}
    
    

    #get the minimum value
    if len(ringV)>0:
        minPos=min(ringV.items(), key=lambda x: x[1]) 
        print("minPos",minPos)

    

    

    state=str(int(X/posdiv))+"_"+str(int(Y/posdiv))
    last_pos_v[state]=total
    #print("STATE: ",state)
    if state in pos:
        pos[state]+=1
    else:
        pos[state]=1
    print("ringV",ringV)
    #print("positions",pos)
    print("SUMA",suma,"TOTAL",total)
    #print("last_pos_v",last_pos_v)

    #get the computed posssible positons with their expected values
    targetpos={}
    for k,v in ringV.items():
        lp=0.1
        if k in last_pos_v:
            lp=last_pos_v[k]
        targetpos[k]=lp/(ringV[k]+0.1)
        
    #print("targetpos",targetpos)

    # pick the lowest value as target
    target_pos = min(targetpos, key=targetpos.get)
    
    print("targetpos",targetpos)
    
    #do the logic to get to target_pos
    if state!=target_pos:
        #if we are not already at target position
        print("target_pos",target_pos,"###############","current",state)
        tX,tY=target_pos.split("_")
        cX,cY=state.split("_")
        if int(tX)>int(cX):
            X+=posdiv/div
        else:
            X-=posdiv/div
        if int(tY)>int(cY):
            Y+=posdiv/div
        else:
            Y-=posdiv/div

    else:
        print("SAME POS:::")
        print("X,Y:",X,Y)
        pass
        """
        X+=(diffX/div)
        Y+=(diffY/div)
        print("diff",diffX,diffY)
        print("X,Y:",X,Y)
        """
    #force max min values
    if X<0:
        print("X limit")
        X=255#10
    if X>255:
        print("X limit")
        X=0#205
    if Y<0:
       print("Y limit bottom")
       Y=170#10
    if Y>170:
        print("Y limit top")
        Y=0#150

    dmx.update_channel(Xchan, int(X))
    dmx.update_channel(Ychan, int(Y))
    dmx.run(interval)
    #sleep(interval)
dmx.close()

