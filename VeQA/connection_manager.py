# -*- coding:utf-8 -*-
from socket import *

class SocketConnection:
    def __init__(self):
        self.clientSocket = socket(AF_INET, SOCK_STREAM)

    def make_connection(self, host='127.0.0.1', port=7770):
        try:
            self.clientSocket.connect((host, port))
        except Exception as e:
            print(e)

    def close_connection(self):
        try:
            self.clientSocket.close()
        except Exception as e:
            print(e)

    def send_message(self, msg, end_mark=False):
        try:
            #self.clientSocket.sendall(bytes(file, 'UTF-8'))
            self.clientSocket.send(msg.encode('utf-8'))
            if end_mark:
                self.clientSocket.send("\n-1\n".encode('utf-8'))
            #print('send file !!')
        except Exception as e:
            print(e)

    def receive_msg(self):
        try:
            msg = self.clientSocket.recv(1024)
            #print(msg)
            msg = msg.decode()
            #print('receive msg !!')
        except Exception as e:
            print(e)

        return msg
