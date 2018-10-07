import socket
import _thread
import time 

def get_sensor_data():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("160.69.69.110",8190))
    s.send(b"Start#")
    while True:
        
        
        msg = s.recv(1024)
        #currentTime = int(round(time.time()*1000))
        print("\n")
        print(msg)
        #prevTime = currentTime
    
    s.close()

    

if __name__ == "__main__":

    get_sensor_data()
