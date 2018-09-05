import socket
import Time


HOST = "192.168.0.9" # The remote host
PORT = 30002 # The same port as used by the server
print "Starting Program"



count = 0

while count==0:
    soc = socket.socket(socket.Af_INET,socket.SOCK_STREAM)
    soc.connect((HOST,PORT))
    time.sleep(0.5)
    soc.send ("set_digital_out(1,True)" + "\n")
    time.sleep(0.1)

    s.send("movej([x,y,z,rx,ry,rj],a=0.1,v=0.1)"+"\n")
    time.sleep(10)

    s.send("movej([x,y,z,rx,ry,rj],a=0.1,v=0.1)"+"\n")
    time.sleep(10)

    s.sendij
