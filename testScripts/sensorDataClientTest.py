import socket

def get_sensor_data():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("160.69.69.110",8190))

    msg  = s.recv(1024)
    s.close()

    print(msg)

if __name__ == "__main__":

    get_sensor_data()
