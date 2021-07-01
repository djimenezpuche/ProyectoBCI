# Desarrollo de un dispositivo (casco) para Brain Computer Interface
El presente proyecto incluye el código correspondiente a un dispositivo para BCI de bajo coste que se ejecutará en una Raspberry Pi y un módulo conversor analógico/digital (A/D).
A través de dicho código se establece la comunicación entre RPi y PC, se realiza la obtención de las señales EEG, se lleva a cabo el procesado de estas,
se calculan las potencias para cada rango de frecuencias EEG y finalmente, se grafican los resultados obtenidos.

- Lenguaje programación empleado: Python3
- Componentes necesarios:
    - RaspberryPi Model 3B (u otra con características similares)
    - Módulo conversión AD/DA (High Precision AD/DA Board) fabricante Waveshare https://www.waveshare.com/wiki/High-Precision_AD/DA_Board

Pasos a seguir:
1. Descargar el código proporcionado por el fabricante Waveshare (https://www.waveshare.com/wiki/File:High-Precision-AD-DA-Board-Code.7z)
2. Descomprimir
3. Accedemos al archivo y vemos que contiene código desarrollado para dos placas computadoras (JetsonNano y RaspberryPi).
4. Accedemos a la carpeta RaspberryPi y vemos que contiene código desarrollado para distintas funcionalidades (AD-DA, ADS1256, DAC8532).
5. En nuestro caso accedemos a ADS1256 y encontramos varios librerías (bcm2835, python2, python3, wiringpi). Mi elección fue Python3.
6. Accedemos a Python3 y vemos tres scripts. (ADS1256.py, config.py, main.py)

El desarrollo de nuestro proyecto comenzó a partir de esos tres scripts. Algunos han sufrido modificaciones y otros son totalmente nuevos.

Los archivos utilizados para el funcionamiento del proyecto son los siguientes:
- ADS1256_ContMode - Desarrolla el funcionamiento del módulo conversor A/D
- config.py - Configuración bus SPI. Este script se importa en el código anterior. 
- rpi_communication.py - Establece la comunicación con PC y realiza la captura de señales EEG a través de las funciones de ADS1256_ContMode
- PC_communication.py - Establece la comunicación con RPi, recoge las señales adquiridas y las procesa mediante diferentes filtros
- power_freqs_processing.py - Realiza el cálculo de las potencias máximas para cada rango de frecuencia y lleva a cabo varias representaciones.

Los tres primeros ( ADS1256_ContMode, config.py y rpi_communication.py) se ejecutan en la RPi. (Colocar en el mismo directorio)
Los dos últimos (PC_communication.py y power_freqs_processing.py) se ejecutan en el PC. (Colocar en el mismo directorio junto con el fichero de metraje del vídeo.)

Ejecución del proyecto:
- Ejecutamos rpi_communication.py en la RPi.
- Ejecutamos PC_communication.py en el PC. Esta ejecución debe incorporar los siguientes parámetros, por ejemplo: metraje_video.txt 720 64
    - Primer parámetro: metraje_video.txt --> Fichero de texto que incluye la temporización del vídeo.
    - Segundo parámetro: 720 --> Duración en segundos del vídeo
    - Tercer parámetro: 64 --> Valor PGA deseado en el ADS1256.
    - Importante destacar que es necesario cambiar el nombre del fichero donde se almacenan los datos para cada experimento.
- A la vez que ejecutamos este último script, debemos iniciar la reproducción del vídeo utilizado en los experimentos con los sujetos.
- Una vez finalizada la adquisición, ejecutamos power_freqs_processing.py. 
      - Importante destacar que es necesario cambiar el nombre del fichero donde se almacenan los datos para cada experimento.


Ficheros adjuntos:
  - Resultados para 5 sujetos. 
  - Ficheros con nombre sujetox.txt --> Datos de salida del script PC_communication.py
  - Ficheros con nombre sujetox_pf.txt --> Datos de salida del script power_freqs_processing.py
  - Vídeo utilizado para los experimentos (Imágenes de los vídeos obtenidas de: https://www.flaticon.es/packs/people-2?k=1612308449169  - Iconos diseñados por            https://www.freepik.com )
