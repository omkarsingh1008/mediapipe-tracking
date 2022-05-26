from re import X
from charset_normalizer import detect
import torch
import cv2

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = model.names
model.to(device)
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width,frame_height)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output = cv2.VideoWriter('output3.mp4', fourcc, fps, frame_size)
x1,y1,x2,y2 = 0,0,0,0
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while True:
        flag = "not pick up"
        _,frame = cap.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame.flags.writeable = False
        results = hands.process(frame)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        frame1 = [frame]
        det = model(frame1)
        #print(det.xyxyn[0])
        labels, cord = det.xyxyn[0][:, -1], det.xyxyn[0][:, :-1]
    
        X = int(x_shape * (50/100))
        start_point = (X,0)
        end_point = (X,y_shape)
        pik=[]
        for i in range(len(cord)):
            bbox=cord[i]
            c=labels[i]
           
            if int(c)==39:
                x1_, y1_, x2_, y2_ = int(bbox[0] * x_shape), int(bbox[1] * y_shape), int(bbox[2] * x_shape), int(bbox[3] * y_shape)
                
                bgr = (0, 255, 0)
                pik.append(x1_)
                cv2.rectangle(frame, (x1_, y1_), (x2_, y2_), bgr, 2)
        for i in pik:
            if i >=X:
                flag = "pick up"


        center_start = (int((start_point[0]+end_point[0])/2),int((start_point[1]+end_point[1])/2))
        center_end = (int(end_point[0]/2),y_shape)

        cv2.putText(img = frame,text = flag,org = (20, 20),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 1.0,color = (125, 246, 55),thickness = 3)
        cv2.line(frame, start_point, end_point, (255,0,0), 2)
        #cv2.line(frame, center_start, center_end, (255,0,0), 2)
        #cv2.circle(frame, center_start, 5, (0,0,0), -1)
        output.write(frame)
        cv2.imshow("frame",frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    output.release()
    cv2.destroyAllWindows()
