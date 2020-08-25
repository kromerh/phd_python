import picamera

camera = picamera.PiCamera()
camera.resolution = (2592,1944)
camera.start_recording('my_video.h264')
camera.wait_recording(10)
camera.stop_recording()