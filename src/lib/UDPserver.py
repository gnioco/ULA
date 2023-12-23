import socket

#Parameters
localPort=8888
bufferSize=1024

#Objects
sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)  ## Internet,UDP


class UDP:
    def __init__(self):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1) #enable broadcasting mode
        sock.bind(('', localPort))
        print("UDP server : {}:{}".format(self.get_ip_address(),localPort))
    
    # function get_ip_address 
    def get_ip_address():
        """get host ip address"""
        ip_address = '';
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8",80))
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address

    def send(self):
           sock.sendto("RPi received OK", addr)  # write data
