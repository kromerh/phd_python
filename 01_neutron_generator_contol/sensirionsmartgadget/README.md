- py3.5 activated!

- sudo apt-get install python-pip libglib2.0-dev
- sudo pip install bluepy
- python3 -m pip install bluepy

- Bluetooth address: C5:BB:A6:86:0E:64 Smart Humigadget
- https://www.elinux.org/RPi_Bluetooth_LE

- gatttool -t random -b C5:BB:A6:86:0E:64 -I
- connect

- TEMP_NOTI_UUID = '00002235-b38d-4985-720e-0f993a68ee41'
- HUMI_NOTI_UUID = '00001235-b38d-4985-720e-0f993a68ee41'
- char-read-uuid 00002235-b38d-4985-720e-0f993a68ee41
