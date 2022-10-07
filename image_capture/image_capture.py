import cv2

# 영상이 있는 경로
vidcap = cv2.VideoCapture('')

count = 0

width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while(vidcap.isOpened()):
    ret, image = vidcap.read()
    # 이미지 사이즈 변경
    image = cv2.resize(image, (width, height))
     
    # 30프레임당 하나씩 이미지 추출
    if(int(vidcap.get(1)) % 30 == 0):
        print('Saved frame number : ' + str(int(vidcap.get(1))))
        # 추출된 이미지가 저장되는 경로
        cv2.imwrite("C:/Users/frame%d.jpg" % count, image)
        #print('Saved frame%d.jpg' % count)
        count += 1
        
vidcap.release()