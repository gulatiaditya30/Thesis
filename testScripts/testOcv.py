import cv2 as cv
import matplotlib.pyplot as plt
import math

def circumfenceCoordinate(Cx,Cy,r):
    i = 0
    

    xArr = []
    yArr = []
    xArr.append(Cx)
    yArr.append(Cy)
    while (i<360):
        

        x = (r*(math.cos(math.radians(i))) + Cx)#*0.001
        y = (r*(math.sin(math.radians(i))) + Cy)#*0.001
        xArr.append(x)
        yArr.append(y)
        i+=1
    print("length is :"+ str(len(xArr)))
    return [xArr,yArr]



if __name__ == "__main__":

    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_EXPOSURE,-120)

    

    while True:    
        ret,frame = cam.read()
        cv.imshow("test",frame)

            
        k = cv.waitKey(1)
        if ("q" == chr(k & 255)):
            exit()

        if ("p" == chr(k & 255)):
            pr = []
            i=0
            while (i<10):
                pr.append(i)
                i=i+1
            print(pr)
            print ("##################################")
            k=0
            while (k<len(pr)):
                print(pr[k])
                k=k+1

        if ("d" == chr(k & 255)):

            a = circumfenceCoordinate(0.0,0.0,2)
            plt.plot(a[0][0],a[1][0],'go')
            plt.plot(a[0][1],a[1][1],'ro')
            plt.plot(a[0][2],a[1][2],'ro')
            '''plt.plot(a[0][3],a[1][3],'ro')
            plt.plot(a[0][4],a[1][4],'ro')
            plt.plot(a[0][5],a[1][5],'ro')
            plt.plot(a[0][6],a[1][6],'ro')'''
            plt.show()
        