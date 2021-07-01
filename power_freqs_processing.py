# -*- coding: utf-8 -*-
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


def normalizar(serie):
    # Procesa en bloques de 103 lecturas equivalente a 250ms.
    # Este proceso gestiona tanto la normalización, 
    # como la deriva ascendente/descendente que presentan 
    # algunos canales.
    # Esta funcion debera ser llamada para cada bloque leido, 
    # antes de enviarse al HOST.
    tam_bloque=412//4
    for base_bloque in range(0,n,tam_bloque):
        bloque=serie[base_bloque:base_bloque+tam_bloque]
        std=np.std(bloque)
        mean=np.mean(bloque)
        minimo=mean-std*3.0
        maximo=mean+std*3.0
        bloque=np.where(bloque>maximo,maximo,bloque)
        bloque=np.where(bloque<minimo,minimo,bloque)
        bloque=(bloque-np.min(bloque))/(np.max(bloque)-np.min(bloque))
        serie[base_bloque:base_bloque+tam_bloque]=bloque
            
    return serie


def power_freqs(signal):
    t=signal.size/412
    sig_fft = fftpack.fft(signal)
    power = np.abs(sig_fft)**2
    sample_freq = fftpack.fftfreq(signal.size, d=1.0/(t*412))
    sample_freq /=t
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    f_power=power[pos_mask]

    p_delta=0.0
    p_theta=0.0
    p_alfa_mu=0.0
    p_beta=0.0
    p_gamma=0.0
    p_ruido=0.0
    for f,p in zip(freqs,f_power):
        if f<=50.0:
            if f>=49.0:
                p_ruido=max(p_ruido,p)
            elif f>=0.5 and f<=3.5:
                p_delta=max(p_delta,p)
            elif f>=3.5 and f<=7.5:
                p_theta=max(p_theta,p)
            elif f>=8 and f<=13:
                p_alfa_mu=max(p_alfa_mu,p)
            elif f>=14.0 and f<=26.0:
                p_beta=max(p_beta,p)
            elif f>=30.0 and f<=45.0:
                p_gamma=max(p_gamma,p)
    return {
        "delta":p_delta,
        "theta":p_theta,
        "alpha_mu":p_alfa_mu,
        "beta":p_beta,
        "gamma":p_gamma,
        "ruido":p_ruido
    }


def power_freqs_block(signal,n_rows):
    # Cálculo de cada rango de frecuencias en bloques de 250ms:
    # Cada número corresponde a la potencia maxima detectada para
    # cada rango de frecuencias en el bloque.
    
    pfs_alpha=[]
    pfs_theta=[]
    pfs_beta=[]
    pfs_gamma=[]
    pfs_delta=[]
    pfs_ruido=[]
    pfs_ch=np.empty([n_rows,0])
    
    for b in range(0,n,412//4):
        pf=power_freqs(signal[b:b+412//4])
        pfs_alpha.append(pf["alpha_mu"])
        pfs_theta.append(pf["theta"])
        pfs_beta.append(pf["beta"])
        pfs_gamma.append(pf["gamma"])
        pfs_delta.append(pf["delta"])
        pfs_ruido.append(pf["ruido"])
    
    pfs_ch=np.insert(pfs_ch,pfs_ch.shape[1],np.array(pfs_alpha),axis=1)
    pfs_ch=np.insert(pfs_ch,pfs_ch.shape[1],np.array(pfs_theta),axis=1)
    pfs_ch=np.insert(pfs_ch,pfs_ch.shape[1],np.array(pfs_beta),axis=1)
    pfs_ch=np.insert(pfs_ch,pfs_ch.shape[1],np.array(pfs_gamma),axis=1)
    pfs_ch=np.insert(pfs_ch,pfs_ch.shape[1],np.array(pfs_delta),axis=1)
    pfs_ch=np.insert(pfs_ch,pfs_ch.shape[1],np.array(pfs_ruido),axis=1)
     
    return pfs_ch

def power_freqs_all_ch(signal,n_rows):
    pf_data=np.empty([n_rows,0])
    for j in range(0,8):
        pf_ch=power_freqs_block(signal[:,j],n_rows)
        pf_data=np.append(pf_data,pf_ch,axis=1) 
    return pf_data

def graphic_power_freqs(signal):
    # ------- Cálculos para el gráfico Power vs Frecuencia señal ----------         
    sig_fft = fftpack.fft(signal)
    power = np.abs(sig_fft)**2
    sample_freq = fftpack.fftfreq(signal.size, d=1.0/(t*412))
    sample_freq /=t
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    f_power=power[pos_mask]
    
    return freqs,f_power
    
#-------------------------------------------------------------------------


signal_matrix=np.loadtxt('sujetoX.txt')
signal_raw=signal_matrix[:,0:8]
signal_autosim=signal_matrix[:,8:16]
signal_butter=signal_matrix[:,16:24]
signal_notch=signal_matrix[:,24:32]
metraje=signal_matrix[:,32]

duration=signal_matrix.shape[0]
n_rows=int(duration/103)
t=duration//412
n=int(412*t)
time = np.linspace(0,t,n)
block=103

# Obtención columna metraje en bloques de 250ms (103 muestras)
metraje_pf=[]

for b in range(0,n,block):
    metraje_block=metraje[b:b+block]
    metraje_pf.append(metraje_block[int((len(metraje_block))//2)])
    
# Señal normalizada
signal_norm=signal_raw.copy()
for i in range (0,8):
    signal_norm[:,i] = normalizar(signal_norm[:,i])
    
# Calculo de potencias en cada rango de frecuencias

pf_raw=power_freqs_all_ch(signal_raw,n_rows)
pf_autosim=power_freqs_all_ch(signal_autosim,n_rows)
pf_butter=power_freqs_all_ch(signal_butter,n_rows)
pf_notch=power_freqs_all_ch(signal_notch,n_rows)

pf_matrix=np.empty([n_rows,0])

pf_matrix=np.append(pf_matrix,pf_raw,axis=1)
pf_matrix=np.append(pf_matrix,pf_autosim,axis=1) 
pf_matrix=np.append(pf_matrix,pf_butter,axis=1) 
pf_matrix=np.append(pf_matrix,pf_notch,axis=1) 
pf_matrix=np.insert(pf_matrix,pf_matrix.shape[1],np.array(metraje_pf),1)

np.savetxt('sujetox_pf.txt',pf_matrix)

# ----------------------- REPRESENTACIONES ---------------------------------

for j in range(0,8):
    
    # ========================= Señal Raw ===================================
    
    # ----------- Representación datos raw de un canal --------------------
    s_raw=signal_raw[:,j]
    
    plt.figure()
    plt.title("Datos raw canal " + str(j) + " - Secuencia completa")
    plt.plot(time[0:duration],s_raw[0:duration])
    plt.show()
    
    plt.figure()
    plt.title("Datos raw canal " + str(j) + " - Secuencia 1 segundo")
    plt.plot(time[82400:82512],s_raw[82400:82512])
    #plt.savefig('signalRaw_1seg.svg',format='svg',dpi=1200)
    plt.show()
    

    
    # ------------ Representación espectro potencia señal raw -------------
    [freqs_raw,f_power_raw] = graphic_power_freqs(s_raw)
    
    plt.figure()
    plt.title("Espectro potencia señal raw - canal " + str(j))
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Power')
    plt.xlim(0,100)
    plt.ylim(0,50)
    plt.plot(freqs_raw,f_power_raw)
    plt.show()
    
    # ===================== Señal normalizada ============================
    
    # ------------- Representación canal normalizado ---------------------
    s_norm=signal_norm[:,j]
    
    plt.figure()
    plt.title("Datos normalizados canal " + str(j) + " - Secuencia completa")
    plt.plot(time[0:412*t],s_norm[0:412*t])
    plt.show()
    
    plt.figure()
    plt.title("Datos normalizados canal " + str(j) + " - Secuencia 1 segundo")
    plt.plot(time[0:412],s_norm[0:412])
    plt.show()
    
    # ------- Representación espectro potencia señal normalizada -----------
    [freqs_norm,f_power_norm] = graphic_power_freqs(s_norm)
    
    plt.figure()
    plt.title("Espectro potencia señal normalizada - canal " +str(j))
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Power')
    plt.xlim(0,100)
    plt.ylim(0,100*1e3)
    plt.plot(freqs_norm,f_power_norm)
    plt.show()
    
    # ===================== Señal Autosimilitud ============================
    
    # ------- Representación espectro potencia señal normalizada -----------
    
    s_autosim=signal_autosim[:,j]
    
    [freqs_autosim,f_power_autosim] = graphic_power_freqs(s_autosim)
    
    plt.figure()
    plt.title("Espectro potencia señal filtrada Autosimilitud - canal " +str(j))
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Power')
    plt.xlim(0,100)
    plt.ylim(0,100*1e3)
    plt.plot(freqs_autosim,f_power_autosim)
    plt.show()
    
    
    # ===================== Señal Butterworth ==============================
    
    # ------- Representación espectro potencia señal normalizada -----------
     
    s_butter=signal_butter[:,j]
    
    [freqs_butter,f_power_butter] = graphic_power_freqs(s_butter)
    
    plt.figure()
    plt.title("Espectro potencia señal filtrada Butterworth - canal " + str(j))
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Power')
    plt.xlim(0,100)
    plt.ylim(0,100*1e3)
    plt.plot(freqs_butter,f_power_butter)
    plt.show()
    
    # ===================== Señal Notch ====================================
    
    # ------- Representación espectro potencia señal normalizada -----------
    s_notch=signal_notch[:,j]
    
    [freqs_notch,f_power_notch] = graphic_power_freqs(s_notch)
    
    plt.figure()
    plt.title("Espectro potencia señal filtrada Notch - canal " + str(j))
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Power')
    plt.xlim(0,100)
    plt.ylim(0,100*1e3)
    plt.plot(freqs_notch,f_power_notch)
    plt.show()
    
    
# =========== Representación por rango de frecuencia para un canal ====================================

id_channel=3 #Indice canal (0,1,2,3,4,5,6,7)
freq_type=0 #Alpha=0,Theta=1,Beta=2,Gamma=3,Delta=4,Ruido=5
filter_type=48 #Raw=0 Autosim=48 Butter=96 Notch=144

filter_name={'Alpha': 0, 'Theta': 1, 'Beta': 2, 'Gamma': 3, 'Delta': 4, 'Ruido': 5}

# k=(6*id_channel+freq_type)+filter_type

# plt.figure()
# plt.title("Potencia señal " + list(filter_name.keys())[list(filter_name.values()).index(freq_type)])
# plt.ylim(0,300)
# plt.plot(pf_matrix[815:815+175,k])
# plt.show()


k1=(6*id_channel+1)+filter_type
k2=(6*id_channel+0)+filter_type
k3=(6*id_channel+2)+filter_type
k4=(6*id_channel+3)+filter_type

fig,ax=plt.subplots(4,1)

ax[0].plot(pf_matrix[815:815+175,k1])
ax[0].set_title('Potencias máximas por bloque')
ax[0].get_xaxis().set_visible(False)
ax[0].set(ylabel='Tetha')

ax[1].plot(pf_matrix[815:815+175,k2])
ax[1].get_xaxis().set_visible(False)
ax[1].set(ylabel='Alfa')

ax[2].plot(pf_matrix[815:815+175,k3])
ax[2].get_xaxis().set_visible(False)
ax[2].set(ylabel='Beta')

ax[3].plot(pf_matrix[815:815+175,k4])
ax[3].set(xlabel='Bloques 250 ms', ylabel='Gamma')
#plt.savefig('p.svg',format='svg',dpi=1200)
plt.show()

# ============ Representación tipo de ondas EEG para un canal =================

s_autosim=signal_autosim[:,3]
bp_alfa= sg.butter(10,[8,13], 'bp', fs=412, output='sos')
alfa=sg.sosfilt(bp_alfa, s_autosim)

s_autosim=signal_autosim[:,3]
bp_beta= sg.butter(10,[14,26], 'bp', fs=412, output='sos')
beta=sg.sosfilt(bp_beta, s_autosim)

s_autosim=signal_autosim[:,3]
bp_gamma= sg.butter(10,[30,45], 'bp', fs=412, output='sos')
gamma=sg.sosfilt(bp_gamma, s_autosim)

s_autosim=signal_autosim[:,3]
bp_theta= sg.butter(10,[3.5,7.5], 'bp', fs=412, output='sos')
theta=sg.sosfilt(bp_theta, s_autosim)

plt.figure()
plt.title("Señal alfa")
plt.xlabel('Tiempo (s)')
plt.ylabel('ΔV (V)')
plt.plot(time[105148:105560],alfa[105148:105560])
#plt.savefig('senal_alfa.svg',format='svg',dpi=1200)
plt.show()

plt.figure()
plt.title("Señal beta")
plt.xlabel('Tiempo (s)')
plt.ylabel('ΔV (V)')
plt.plot(time[105148:105560],beta[105148:105560])
#plt.savefig('senal_beta.svg',format='svg',dpi=1200)
plt.show()

plt.figure()
plt.title("Señal gamma")
plt.xlabel('Tiempo (s)')
plt.ylabel('ΔV (V)')
plt.plot(time[105148:105560],gamma[105148:105560])
#plt.savefig('senal_gamma.svg',format='svg',dpi=1200)
plt.show()

plt.figure()
plt.title("Señal theta")
plt.xlabel('Tiempo (s)')
plt.ylabel('ΔV (V)')
plt.plot(time[105148:105560],theta[105148:105560])
#plt.savefig('senal_theta.svg',format='svg',dpi=1200)
plt.show()



# # =====================  Representaciones individuales  (para un canal)======================

# # ----------- Representación datos raw de un canal --------------------

s_raw=signal_raw[:,3]
plt.figure()
plt.title("Datos raw canal " + str(3) + " - Secuencia completa")
plt.xlabel('Tiempo (s)')
plt.ylabel('ΔV (V)')
plt.plot(time[4:duration],s_raw[4:duration])
#plt.savefig('Raw_64_Canal3_bp.svg',format='svg',dpi=1200)
plt.show()

s_raw=signal_raw[:,3]
plt.figure()
plt.title("Datos raw canal " + str(3) + " - Secuencia 1 segundo")
plt.xlabel('Tiempo (s)')
plt.ylabel('ΔV (V)')
plt.plot(time[105148:105560],s_raw[105148:105560])
#plt.savefig('Raw_64_Canal3_1seg_bp.svg',format='svg',dpi=1200)
plt.show()

s_raw=signal_raw[:,3]
[freqs_raw,f_power_raw] = graphic_power_freqs(s_raw)
plt.figure()
plt.title("Espectro potencia señal raw - canal " + str(3))
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Power')
plt.xlim(0,100)
plt.ylim(0,200)
plt.plot(freqs_raw,f_power_raw)
#plt.savefig('EP_Raw_16_Canal3_bp.svg',format='svg',dpi=1200)
plt.show()

# # ----------- Representación señal normalizada de un canal --------------------

s_norm=signal_norm[:,3]

plt.figure()
plt.title("Datos normalizados canal " + str(3) + " - Secuencia completa")
plt.xlabel('Tiempo (s)')
plt.ylabel('ΔV (V)')
plt.plot(time[0:412*t],s_norm[0:412*t])
#plt.savefig('Raw_64_Canal3_norm_bp.svg',format='svg',dpi=1200)
plt.show()

s_norm=signal_norm[:,3]
plt.figure()
plt.title("Datos normalizados canal " + str(3) + " - Secuencia 1 segundo")
plt.xlabel('Tiempo (s)')
plt.ylabel('ΔV (V)')
plt.plot(time[105148:105560],s_norm[105148:105560])
#plt.savefig('Raw_64_Canal3_norm_1seg_bp.svg',format='svg',dpi=1200)
plt.show()

s_norm=signal_norm[:,3]
[freqs_norm,f_power_norm] = graphic_power_freqs(s_norm)
    
plt.figure()
plt.title("Espectro potencia señal normalizada - canal " +str(3))
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Power')
plt.xlim(0,100)
plt.ylim(0,500*1e4)
plt.plot(freqs_norm,f_power_norm)
#plt.savefig('EP_norm_64_Canal3_bp.svg',format='svg',dpi=1200)
plt.show()

s_autosim=signal_autosim[:,3]
plt.figure()
plt.title("Señal filtro autosimilitud canal " + str(3) + " - Secuencia completa")
plt.xlabel('Tiempo (s)')
plt.ylabel('ΔV (V)')
plt.plot(time[0:duration],s_autosim[0:duration])
#plt.savefig('Autosim_64_Canal3_bp.svg',format='svg',dpi=1200)
plt.show()

s_autosim=signal_autosim[:,3]
plt.figure()
plt.title("Señal filtro autosimilitud canal " + str(3) + " - Secuencia 1 segundo")
plt.xlabel('Tiempo (s)')
plt.ylabel('ΔV (V)')
plt.plot(time[105148:105560],s_autosim[105148:105560])
#plt.savefig('Autosim_64_Canal3_1seg_bp.svg',format='svg',dpi=1200)
plt.show()

s_autosim=signal_autosim[:,3]

[freqs_autosim,f_power_autosim] = graphic_power_freqs(s_autosim)

plt.figure()
plt.title("Espectro potencia señal filtrada Autosimilitud - canal " +str(3))
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Power')
plt.xlim(0,100)
plt.ylim(0,500*1e6)
plt.plot(freqs_autosim,f_power_autosim)
#plt.savefig('EP_Autosim_64_Canal3_bp.svg',format='svg',dpi=1200)
plt.show()

# # ----------- Representación señal butter de un canal --------------------

s_butter=signal_butter[:,3]

plt.figure()
plt.title("Señal filtro Butterworth canal " + str(3) + " - Secuencia 1 segundo")
plt.xlabel('Tiempo (s)')
plt.ylabel('ΔV (V)')
plt.plot(time[105148:105560],s_autosim[105148:105560])
#plt.savefig('Butter_64_Canal3_1seg_bp.svg',format='svg',dpi=1200)
plt.show()


s_butter=signal_butter[:,3]
[freqs_butter,f_power_butter] = graphic_power_freqs(s_butter)

plt.figure()
plt.title("Espectro potencia señal filtrada Butterworth - canal " + str(3))
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Power')
plt.xlim(0,100)
plt.ylim(0,500*1e6)
plt.plot(freqs_butter,f_power_butter)
#plt.savefig('EP_Butter_64_Canal3_bp.svg',format='svg',dpi=1200)
plt.show()

# # ----------- Representación señal notch de un canal --------------------

s_notch=signal_notch[:,3]

plt.figure()
plt.title("Señal filtro Notch canal " + str(3) + " - Secuencia 1 segundo")
plt.xlabel('Tiempo (s)')
plt.ylabel('ΔV (V)')
plt.plot(time[105148:105560],s_autosim[105148:105560])
#plt.savefig('Notch_64_Canal3_1seg_bp.svg',format='svg',dpi=1200)
plt.show()

s_notch=signal_notch[:,3]
[freqs_notch,f_power_notch] = graphic_power_freqs(s_notch)

plt.figure()
plt.title("Espectro potencia señal filtrada Notch - canal " + str(3))
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Power')
plt.xlim(0,100)
plt.ylim(0,500*1e6)
plt.plot(freqs_notch,f_power_notch)
#plt.savefig('EP_Notch_64_Canal3_bp.svg',format='svg',dpi=1200)
plt.show()


# ========== Representación espectro potencia 3 filtros superpuestos =========

s_autosim=signal_autosim[:,3]
[freqs_autosim,f_power_autosim] = graphic_power_freqs(s_autosim)

s_notch=signal_notch[:,3]
[freqs_notch,f_power_notch] = graphic_power_freqs(s_notch)

s_butter=signal_butter[:,3]
[freqs_butter,f_power_butter] = graphic_power_freqs(s_butter)

plt.figure()
plt.title("Espectro potencia señal filtrada - canal " +str(3))
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Power')
plt.xlim(0,100)
plt.ylim(0,500*1e6)
plt.plot(freqs_autosim,f_power_autosim,color='green',label='autosim')
plt.plot(freqs_butter,f_power_butter,color='red',label='butter')
plt.plot(freqs_notch,f_power_notch,color='blue',label='notch')

plt.legend(loc='upper right')

#plt.savefig('EP_Filtros_64_Canal3_bp.svg',format='svg',dpi=1200)
plt.show()


