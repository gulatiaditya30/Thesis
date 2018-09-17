import cv2 as cv

if __name__ == "__main__":

    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_EXPOSURE,-120)

    

    while True:    
        ret,frame = cam.read()
        cv.imshow("test",frame)
        
        k = cv.waitKey(1)
        if ("q" == chr(k & 255)):
            exit()

        