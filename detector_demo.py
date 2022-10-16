import torch
import os
import cv2
import matplotlib.pyplot as plt

model_path = './vehicle_detector/best.pt'
model = torch.hub.load(os.getcwd(), 'custom', source='local', path = model_path, force_reload=True)
video = './video/test2_Trim.mp4'

cap = cv2.VideoCapture(video)
if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            result = model(img)
            fig, ax = plt.subplots(figsize=(16,12))
            ax.imshow(result.render()[0])
            plt.show()
