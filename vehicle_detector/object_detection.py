import torch
import cv2
import numpy as np
from time import time

class Detector():
    def __init__(self, video):
        self.model = self.load_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.video = video

    def load_model(self):
        model = torch.load('../1st/best.pt')

        return model
    
    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()

        return labels, cord
    
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cord[i]

            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0,255,0)
                cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i])
                            + ': ' + str(x1) + ', ' + str(x2) + ', ' + str(y1) + ', ' + str(y2),
                            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                
        return frame

    def __call__(self):
        # 인스턴스 생성 시 호출; 프레임 단위로 비디오 로드
        cap = cv2.VideoCapture(self.video)
        
        while True:
            start_time = time()
            ret, frame = cap.read()
            if not ret:
                print("No video")
                break

            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)
            print(f"Frames Per Second : {fps}")

            cv2.imshow(frame)

if __name__ == '__main__':
    video = './video/test2_Trim.mp4'
    Execution = Detector(video)
    Execution()