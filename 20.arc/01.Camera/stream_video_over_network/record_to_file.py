import picamera

camera = picamera.PiCamera()
camera.resolution = (800,600)
camera.color_effects = (128,128)
camera.start_recording('my_video.h264')
camera.wait_recording(10)
camera.stop_recording()