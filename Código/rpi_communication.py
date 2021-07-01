## ---------- RPi_Communication ----------- ##

##!/usr/bin/env python

import time
import ADS1256_ContMode
import RPi.GPIO as GPIO
import io
import pickle
import sys
import json
import numpy as np
from socket import *
import struct

def send_msg(sock,msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg=struct.pack('>I',len(msg))+msg
    sock.send(msg)



#Configuracion socket
host = ''
port_udp = 12000
port_tcp=14008

s = socket(AF_INET, SOCK_DGRAM)
s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
s.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
s.bind((host,port_udp))

print("Listening ...")

while True:
    ##print("Dentro del bucle")
    msg, (pc_addr, port) = s.recvfrom(1024)
    message=msg.decode("utf-8")
    print(message)
    if message == "FINDING BCI":
        print("BCI found")
        print("Direccion PC recibida: ", pc_addr)
        m = "OK BCI"
        s.sendto(m.encode("utf-8"),(pc_addr,port))
        break

print("fin")
s.close()

#------------------------- Socket TCP para intercambio de datos eeg ------------------
s_tcp = socket(AF_INET, SOCK_STREAM)
s_tcp.connect((pc_addr,port_tcp))

try:
    ADC = ADS1256_ContMode.ADS1256()
    ADC.ADS1256_init()

    ## TEST INDICANDO NUM DE MUESTRAS - Obtenemos tiempo de adquisicion y muestras (deben ser las mismas que pasamos por parametro)

    duration = 296640
    block = 103
    
#    ADC_matrix=[]
    start=time.time()
    for i in range(duration//block):
        ADC_matrix=[]
        for v in range(block):
            ADC_Value = ADC.ADS1256_GetAll()
            ADC_matrix.append(ADC_Value)
        data=pickle.dumps(ADC_matrix)
        send_msg(s_tcp,data)
    s_tcp.close()
    execution_time=time.time()-start
    print("Tiempo ejecucion " + str(execution_time))
    
 

except:
    GPIO.cleanup()
    #f.close()
    #print("Duration: " + str(duration))
    #print(len(data))
    #s_tcp.close()
    print ("\r\nProgram end")
    exit()