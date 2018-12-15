import numpy as np
import cv2 

cap = capture =cv2.VideoCapture('modis_data/afg_full_time/final_boss.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()


    cv2.imshow('frame',frame)
    cv2.waitKey(24)

cap.release()
cv2.destroyAllWindows()
