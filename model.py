import numpy as np
import cv2
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import chisquare

def spatial_histogram(src, m):
    height, width = src.shape
    # Take size of block;
    h_blocks = int(height / m)
    w_blocks = int(width / m)
    H = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            H[i, j] = np.sum(src[(i * h_blocks):((i + 1) * h_blocks - 1), (j * w_blocks):((j + 1) * w_blocks - 1)])
    return H


def diagonal_matrix(matrix):
    diagonal = np.zeros(shape=(N + K, N + K))
    row, col = matrix.shape
    for i in range(row):
        diagonal[i, i] = sum(matrix[:, i])

    return diagonal

file_path = "dataset/video1.mp4"
cap = cv2.VideoCapture(file_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = int(frameCount / fps)

# Take the first frame and convert it to gray
ret, frame1 = cap.read()
gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Create the HSV color image
hsvImg = np.zeros_like(frame1)
print(hsvImg.shape)
hsvImg[..., 1] = 255
data = []

while True:
    # Save the previous frame data
    previousGray = gray

    # Get the next frame
    ret, frame2 = cap.read()
    if ret:
        # Convert the frame to gray scale
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate the dense optical flow
        flow = cv2.calcOpticalFlowFarneback(previousGray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Obtain the flow magnitude and direction angle
        magnitude, direction_angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # 		magnitude = magnitude / np.max(magnitude)
        H = spatial_histogram(magnitude, 10)
        H = H.ravel()
        data.append(H)
    # Update the color image
    # hsvImg[..., 0] = 0.5 * direction_angle * 180 / np.pi
    # hsvImg[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # rgbImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)

    # Display the resulting frame
    # imS = cv2.resize(np.hstack((frame2, rgbImg)), (960, 540))
    # plt.imshow(H)
    # plt.imshow(imS)
    # plt.show()

    # Exit if the user press ESC
    if cv2.waitKey(1) and ret == False:
        break

# When everything is done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

for item in data:
    for i in item:
        if np.isinf(i) == True:
            i = 0

#find K prototype
K = 500
kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
center = kmeans.cluster_centers_
np.savetxt('Kmeans_cluster_centers.txt', kmeans.cluster_centers_, delimiter=',')
center = np.float32(center)

predict_kmean = kmeans.predict(data)

#slide video
overlap_frame = 30
frame_per_segment = 120
start_frame = 0

N = math.ceil((frameCount - overlap_frame)/(frame_per_segment - overlap_frame))
C = np.zeros(shape=(N,K))
segment_part=[]

for j in range(0,N):
    stop_frame = start_frame + frame_per_segment - 1
    for i in range(start_frame, stop_frame):
        cap.read()
        x = int(predict_kmean[i])
        C[j,x] = 1
        if frameCount < stop_frame:
            stop_frame = frameCount-1
            break
    segment_part.append([start_frame,stop_frame])
    start_frame = stop_frame - overlap_frame
    cap.set(cv2.CAP_PROP_FRAME_COUNT,start_frame)
    cap.set(cv2.CAP_PROP_FRAME_COUNT,start_frame)

S = np.zeros(shape=(K,K))
for i in range(K):
    for j in range(K):
#         S[i,j] = (cv2.compareHist(np.reshape(center[i],(10,10)), np.reshape(center[j],(10,10)),cv2.HISTCMP_CHISQR) +
#                 cv2.compareHist(np.reshape(center[j],(10,10)), np.reshape(center[i],(10,10)),cv2.HISTCMP_CHISQR))/2
        S[i,j] = (chisquare(center[i],center[j])[0] + chisquare(center[j],center[i])[0])/2

#Calculate matrix W:
# Snorm = np.uint8(S)
belta = 1/np.max(S)
# Snorm = np.float32(S)
W = np.eye(N, dtype = float)
C_T = C.transpose()
W = np.append(W,C,axis=1)
W = np.append(W,np.append(C_T,belta*S,axis=1),axis=0)

# plt.imshow(S)
# plt.show()
# print(np.max(W))
# print(np.min(W))
# print(np.min(S))
# print(np.max(S))

#Optimal problem
D = diagonal_matrix(W)
lamda, eig_vector = np.linalg.eigh((D-W).dot(np.linalg.inv(D)))
x = eig_vector[:,1]
# plt.scatter(x,np.zeros(K+N)+10)
# plt.show()

kmean2 = KMeans(n_clusters=2, random_state=0).fit(x.reshape(-1,1))
center2 = kmean2.cluster_centers_

#find the most diffirent cluster
dists = euclidean_distances(kmean2.cluster_centers_)
total = np.sum(dists,axis=0) #total of inter-cluster distance
abnormal_cluster = np.argmax(total)

#Show result
labels_segment = kmean2.predict(x.reshape(-1,1))[:N]
idx = [i for i, j in enumerate(labels_segment) if j == abnormal_cluster]
segment_abnormal = []
for i in idx:
    segment_abnormal.append(segment_part[i])

predict = np.zeros(shape=frameCount)
for i in range(frameCount):
    for j in range(len(idx)):
        if segment_abnormal[j][0] <= i and i <= segment_abnormal[j][1]:
            predict[i] = 1

frame_abnormal = np.asarray([i for i, j in enumerate(predict) if j == 1])
ground_truth = np.zeros(shape=frameCount)
for i in range(frameCount):
    if (500<=i and i<=600) or (1305<=i and i<=1432):
#     if (490<=i and i<=600) or (1320<=i and i<=1420) or (1790<=i and i<=1964):
        ground_truth[i] = 1

#show the result frame
# num_frame = [i for i in range(frameCount)]
#
# fig, ax = plt.subplots()
# ax.plot(num_frame, predict, label="PREDICT")
# ax.plot(num_frame, ground_truth, label="GROUND_TRUTH")
# ax.legend()
#
# plt.show()
box_predict = []
j = 0
left = frame_abnormal[j]
while True:
    if frame_abnormal[j+1] - frame_abnormal[j] != 1:
        right = frame_abnormal[j]
        box_predict.append([left,right])
        left = frame_abnormal[j+1]
    j = j + 1
    if j == (len(frame_abnormal) - 1):
        right = frame_abnormal[-1]
        box_predict.append([left,right])
        break

box_ground = [[500,600],[1305,1432]]

def box_iou(pred, ground, num_pred, num_ground):
    pred_left = pred[num_pred][0]
    pred_right = pred[num_pred][1]
    ground_left = ground[num_ground][0]
    ground_right = ground[num_ground][1]

    edge_left = np.min([pred_left,pred_right,ground_left,ground_right]) #0
    edge_right = np.max([pred_left,pred_right,ground_left,ground_right]) #600

    pred_list = np.zeros(shape=frameCount)
    ground_list = np.zeros(shape=frameCount)
    for y in range(len(pred_list)):
        if pred_left <= y and y <= pred_right:
            pred_list[y] = 1  # 0->475: 1
        elif ground_left <= y and y <= ground_right:
            ground_list[y] = 1  # 500 -> 600 : 1

    intersection = np.logical_and(pred_list, ground_list).sum()  # 25
    union = edge_right - edge_left - intersection
    iou = intersection / union

    return iou

for i in range(len(box_predict)):
    for j in range(len(box_ground)):
        iou = box_iou(box_predict,box_ground,i,j)
        print(iou)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('result.avi',-1, 20, (frame_width,frame_height))
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        if predict[i] == 1:
            cv2.putText(frame,'Abnormal activity',(10,220),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_8, False)
        out.write(frame)
        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        i = i + 1
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()