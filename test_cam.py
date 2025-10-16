from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.start()
frame = picam2.capture_array()
print("Frame shape:", frame.shape)
cv2.imwrite("test.jpg", frame)
