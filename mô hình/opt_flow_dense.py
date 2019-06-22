import numpy as np
import cv2
import matplotlib.pyplot as plt


file_path = "dataset/video7.mp4"

cap = cv2.VideoCapture(file_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = int(frameCount / fps)
# print('fps = ' + str(fps))
# print('number of frames = ' + str(frameCount))
# print('duration (S) = ' + str(duration))

# Take the first frame and convert it to gray
ret, frame1 = cap.read()
gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Create the HSV color image
hsvImg = np.zeros_like(frame1)
hsvImg[..., 1] = 255
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('dense_optical_flow1.avi',fourcc, 10, (frame_width*2, frame_height*2))
while True:
    # Save the previous frame data
    previousGray = gray
    
    # Get the next frame
    ret, frame2 = cap.read()
    if ret:
        # Convert the frame to gray scale
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(3,3),1.5)
        # Calculate the dense optical flow
        flow = cv2.calcOpticalFlowFarneback(previousGray, gray, None, 0.5, 1, 5, 3, 3, 2.5, 0)
        
        # Obtain the flow magnitude and direction angle
        magnitude, direction_angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # print(np.max(magnitude))
        magnitude = magnitude / 100
    # Update the color image
        hsvImg[..., 0] = 0.5 * direction_angle * 180 / np.pi
        hsvImg[..., 2] = cv2.normalize(magnitude, None, 255, 0, cv2.NORM_MINMAX)
        rgbImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    
    # Display the resulting frame
        imS = cv2.resize(np.hstack((frame2, rgbImg)), (960, 540))
        out.write(rgbImg)
    # plt.imshow(H)
        cv2.imshow("sample",imS)
    
    # Exit if the user press ESC
    if cv2.waitKey(1) and ret == False:
        break

# When everything is done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()