import cv2
import torch
import matplotlib.pyplot as plt
import os

model_path = './vehicle_detector/best.pt'
model = torch.hub.load(os.getcwd(), 'custom', source='local', path = model_path, force_reload=True)
video = './video/test2_Trim.mp4'
print(torch.cuda.is_available())

cap = cv2.VideoCapture(video)
if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            results = model(img)
            print("----- results -----")
            print(results)

            # Number of cars in current frame
            num_of_cars = len(results.pandas().xyxy[0]['xmin'])

            # Lists of results from yolo directly
            xmin_list = []
            ymin_list = []
            xmax_list = []
            ymax_list = []
            confidence_list = []

            # Lists of width, height, centroid calculation
            width_list = []
            height_list = []
            centroid_list = []

            # Add into lists & draw bbox
            for iter in range(num_of_cars):
                print("----- ", iter, "st car ------")
                xmin_list.append(results.pandas().xyxy[0]['xmin'][iter])
                ymin_list.append(results.pandas().xyxy[0]['ymin'][iter])
                xmax_list.append(results.pandas().xyxy[0]['xmax'][iter])
                ymax_list.append(results.pandas().xyxy[0]['ymax'][iter])
                confidence_list.append(results.pandas().xyxy[0]['confidence'][iter])

                print("xmin : ", results.pandas().xyxy[0]['xmin'][iter])
                print("ymin : ", results.pandas().xyxy[0]['ymin'][iter])
                print("xmax : ", results.pandas().xyxy[0]['xmax'][iter])
                print("ymax : ", results.pandas().xyxy[0]['ymax'][iter])
                print("confidence : ", results.pandas().xyxy[0]['confidence'][iter])

                width_list.append(abs(int(xmax_list[iter]) - int(xmin_list[iter])))
                height_list.append(abs(int(ymax_list[iter]) - int(ymin_list[iter])))
                centroid_list.append(((int(xmin_list[iter]) + int(xmax_list[iter]))/2, (int(ymin_list[iter]) + int(ymax_list[iter]))/2))

                # Check
                # print(type(confidence_list[iter]))

                # cv2.putText(img, str(confidence_list[iter]), (int(xmin_list[iter]), int(ymin_list[iter])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
                # cv2.rectangle(img, (int(xmin_list[iter]), int(ymin_list[iter])), (int(xmax_list[iter]), int(ymax_list[iter])), (0, 0, 255), 3)                

            # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # plt.imshow(imgRGB)
            # plt.show()

            cv2.imshow('Result', img)
        
        else:
            break
else:
    print("can't open video")

cap.release()
cv2.destroyAllWindows()