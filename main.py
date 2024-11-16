from time import sleep
from curiosity import curiosity
from camera import camera

cameraindex=2
CAM=camera(cameraindex=cameraindex,preview=True)

CAM.start_cam()
sleep(5)
CST=curiosity(CAM,pause=0)
CST.start()

running=True

while running:
    
    sides={0:"topleft",1:"topright",2:"bottomleft",3:"bottomright"}
    #maxindex=index_min = max(range(len(CST.state_vals)), key=CST.state_vals.__getitem__)
    maxv=max(CST.state_vals)
    maxindex=CST.state_vals.index(maxv)
    median=sum(CST.state_vals)/4
    maxside=sides[maxindex]
    ratio=maxv-median
    print(maxside,ratio,CST.state_vals)
    #print()
    sleep(0.1)