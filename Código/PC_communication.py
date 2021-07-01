#----------------------------- PC_Communication ------------------------------

# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
@author: Diego Jiménez Puche
"""

import sys
import numpy as np
import pylab as plt
from scipy.fftpack import rfft, irfft, fftfreq,fft
from scipy import fftpack
from scipy import signal as sg
import matplotlib.pyplot as matplt
from socket import *
import select
import json
import pickle
import struct

def normalizar(bloque):
    # Procesa en bloques de 103 lecturas = 412/4
    # 103 lecturas equivalen a 250 milisegundos.
    # Este proceso gestiona tanto la normalización, 
    # como la deriva ascendente/descendente que presentan 
    # algunos canales.
    # Esta funcion debera ser llamada para cada bloque leido, 
    # antes de enviarse al HOST.
    std=np.std(bloque)
    mean=np.mean(bloque)
    minimo=mean-std*3.0
    maximo=mean+std*3.0
    bloque=np.where(bloque>maximo,maximo,bloque)
    bloque=np.where(bloque<minimo,minimo,bloque)
    bloque=(bloque-np.min(bloque))/(np.max(bloque)-np.min(bloque))
    return bloque

def filter_autosim(signal_patrones):
    # "patrones" es la señal de 50Hz calculada usando 
    # la propia serie a filtrar. Gracias a la autosimilitud.
        
    patrones=np.zeros(len(signal_patrones))
    patrones[0:16]=np.divide(np.add(signal_patrones[0:16],signal_patrones[16:32],signal_patrones[32:48]),3)
    patrones[block-16:block]=np.divide(np.add(signal_patrones[block-16:block],signal_patrones[block-32:block-16],signal_patrones[block-48:block-32]),3)
    for x in range(16,block-16):
     patrones[x]=(signal_patrones[x-16]+signal_patrones[x-8]+signal_patrones[x+8]+signal_patrones[x+16])/4
     
    # Esta linea, permite calcular mean usando la media de todos
    # los canales. Funciona mejor la autosimilitud por lo que es mejor dejar comentada.
    #mean=np.median(chs[:,0:7],axis=1)
    mean=patrones
    
    # La idea es restar a la señal, la calculada para una
    # interferencia de 50Hz. Así, se elimina en gran parte
    # en ruido a dicha frecuencia.
    # Cuando mas eficaz sea el filtro mas se solapara en el 
    # grafico 4, la linea verde (señal) con el ruido estimado.
    # Ademas, cuanto mejor sea el filtro mejor saldra en la figura 2
    # la potencia de señales en los rangos buscados.
    signal_patrones=np.subtract(signal_patrones,mean)
    
    return signal_patrones

def filter_butter(signal_butter):
    # Filtro Butterworth configurado como bandstop  
    fs_butter=412 #Frecuencia de muestreo. A pesar de configurar el muestreo a 30kHz obtenemos aprox 412 muestras/seg
    low = 49
    high = 51
    N_order=10
    butter_filter= sg.butter(N_order,[low,high], 'bs', fs=fs_butter, output='sos')
    signal_butter=sg.sosfilt(butter_filter, signal_butter)
    return signal_butter
    
def filter_notch(signal_notch):
    # #Filtro notch
    fs_notch = 412 # Sample frequency (Hz)
    f0 = 50 # Frequency to be removed from signal (Hz)
    Q = 10   # Quality factor
    b, a = sg.iirnotch(f0, Q, fs_notch)
    signal_notch = sg.filtfilt(b, a, signal_notch)
    return signal_notch

def recv_msg(sock):
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    return recvall(sock, msglen)

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n-len(data))
        if not packet:
            return None
        data.extend(packet)
    return data
    

## ------------------------------ Lectura de parametros ----------------------

nombre_metraje = sys.argv[1] #Nombre del archivo con el metraje del video
duracion = int(sys.argv[2])*412 #Recibe duracion en segundos y convierte a muestras
pga = sys.argv[3] #Valor PGA

block=103             # Tamano bloque muestras
N_ROWS=duracion       # Número máximo de muestras a leer.
N_CHS=8               # Número de canales.


f_metraje = open(nombre_metraje,"r")
command=[]
for i in f_metraje:
    v=i.split(",")
    for j in range(int(v[0]),int(v[1])):
        command.append(float(v[2]))

## ------------------------------- Conexión con Rpi -------------------------

host= '<broadcast>'
port_udp= 12000
port_tcp= 14008


s = socket(AF_INET, SOCK_DGRAM)
s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
s.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
s.setblocking(0)

print("Connecting...")

rpi_addr=""

while True:
    msg="FINDING BCI"
    s.sendto(msg.encode("utf-8"),(host,port_udp))
    exist_data = select.select([s],[],[],0.5)
    if exist_data[0]:
        msg_rcv,(address,p) = s.recvfrom(1024)
        message=msg_rcv.decode("utf-8")
        print(message)
        if message =="OK BCI":
            rpi_addr=address
            #s.sendto(.encode("utf-8"),(rpi_addr,port_udp)) #mandar duracion y PGA
            #print(rpi_addr)
            print("fin bucle")
            break
    
s.close()

# ------------ Socket TCP para intercambio de datos EEG ---------------
#try:
s_tcp = socket(AF_INET, SOCK_STREAM)
s_tcp.bind(('',port_tcp))
s_tcp.listen(1)
#s_tcp.setblocking(1)

print("Socket TCP Ok")

con, addr = s_tcp.accept()

raw_matrix=np.empty([0,N_CHS])
autosim_matrix=np.empty([0, N_CHS])
butter_matrix=np.empty([0, N_CHS])
notch_matrix=np.empty([0, N_CHS])
final_signal_matrix=np.empty([N_ROWS, 0])

m=0

while m<duracion//block:
    
    # Recibimos los datos por el socket
    data=recv_msg(con)
    data_dec=pickle.loads(data)
    block_raw=np.array(data_dec,dtype=float)
    raw_matrix=np.append(raw_matrix,block_raw,axis=0)
        
    #Normalizamos
    chs=block_raw.copy()
    
    for c in range(0,N_CHS):
        chs[:,c]=normalizar(block_raw[:,c])
    
    #Filtramos
    autosim_matrix_block=np.empty([block, N_CHS])
    butter_matrix_block=np.empty([block, N_CHS])
    notch_matrix_block=np.empty([block, N_CHS])
    
    
    for chan in range(0,N_CHS):    
        signal_patrones=chs[:,chan]
        signal_butter=chs[:,chan]
        signal_notch=chs[:,chan]
               
        # # ------------------ PATRONES ---------------------------
        signal_patrones=filter_autosim(signal_patrones)
        
    
        # ------------------ BUTTERWORTH -----------------------------------------
        # Filtro Butterworth configurado como bandstop
        signal_butter=filter_butter(signal_butter)
        
        
        # ------------------ NOTCH -----------------------------------------------
        # #Filtro notch
        signal_notch=filter_notch(signal_notch)
            
        #Incluimos en cada iteración el canal filtrado en su matriz correspondiente
        autosim_matrix_block[:,chan]=signal_patrones
        butter_matrix_block[:,chan]=signal_butter
        notch_matrix_block[:,chan]=signal_notch
    
    autosim_matrix=np.append(autosim_matrix,autosim_matrix_block,axis=0)
    butter_matrix=np.append(butter_matrix,butter_matrix_block,axis=0)
    notch_matrix=np.append(notch_matrix,notch_matrix_block,axis=0)
    
    m=m+1

final_signal_matrix=np.append(final_signal_matrix,raw_matrix,axis=1)
final_signal_matrix=np.append(final_signal_matrix,autosim_matrix,axis=1)
final_signal_matrix=np.append(final_signal_matrix,butter_matrix,axis=1)
final_signal_matrix=np.append(final_signal_matrix,notch_matrix,axis=1)
final_signal_matrix=np.insert(final_signal_matrix, final_signal_matrix.shape[1],command,1)

np.savetxt('sujetoX.txt',final_signal_matrix,'%1.6f')
    
con.close()
s_tcp.close()
print("fin")