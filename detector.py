import cv2
import torch
import matplotlib.pyplot as plt
import os
import timeit


model_path = './vehicle_detector/best.pt'
model = torch.hub.load(os.getcwd(), 'custom', source='local', path = model_path, force_reload=True)
video = './video/test1.mp4'
print(torch.cuda.is_available())

cap = cv2.VideoCapture(video)
if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            start_t = timeit.default_timer()
            results = model(img)
            # print("----- results -----")
            # print(results)

            # Number of cars in current frame
            num_of_cars = len(results.pandas().xyxy[0]['xmin'])

            # Lists of results from yolo directly
            # xmin_list = []
            # ymin_list = []
            # xmax_list = []
            # ymax_list = []
            # confidence_list = []

            # Lists of width, height, centroid calculation
            # width_list = []
            # height_list = []
            # centroid_list = []

            # # Add into lists & draw bbox
            for iter in range(num_of_cars):
                # print("----- ", iter, "st car ------")
                xmin = (results.pandas().xyxy[0]['xmin'][iter])
                ymin = (results.pandas().xyxy[0]['ymin'][iter])
                xmax = (results.pandas().xyxy[0]['xmax'][iter])
                ymax = (results.pandas().xyxy[0]['ymax'][iter])
                confidence = (results.pandas().xyxy[0]['confidence'][iter])

                # print("xmin : ", xmin)
                # print("ymin : ", ymin)
                # print("xmax : ", xmax)
                # print("ymax : ", ymax)
                # print("confidence : ", confidence)

                # width_list.append(abs(int(xmax) - int(xmin)))
                # height_list.append(abs(int(ymax) - int(ymin)))
                # centroid_list.append(((int(xmin) + int(xmax))/2, (int(ymin) + int(ymax))/2))

                cv2.putText(img, str(confidence), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 5)                

            # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # plt.imshow(imgRGB)
            # plt.show()
            end_t = timeit.default_timer()
            fps = int(1./(end_t - start_t))
            print("FPS : ", fps)
            cv2.imshow('Result', img)
            cv2.waitKey(1)
            
        
        else:
            break
else:
    print("can't open video")

cap.release()
cv2.destroyAllWindows()