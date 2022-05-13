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
x1,y1,x2,y2 = 0,0,0,0
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while True:
        flag = "pick up"
        _,frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(frame)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        frame1 = [frame]
        det = model(frame1)

        labels, cord = det.xyxyn[0][:, -1], det.xyxyn[0][:, :-1]
            
        p = y_shape * (20/100)
        y1 = y_shape - p
        x2 = x_shape
        y2 = y1
        start_point = (x1,int(y1))
        end_point = (x2,int(y2))
        pik=[]
        for i in range(len(cord)):
            bbox=cord[i]
            c=labels[i]
            print(labels[i])
            print(classes[int(c)])
            if int(c)==39:
                x1_, y1_, x2_, y2_ = int(bbox[0] * x_shape), int(bbox[1] * y_shape), int(bbox[2] * x_shape), int(bbox[3] * y_shape)
                bgr = (0, 255, 0)
                pik.append(y2_)
                cv2.rectangle(frame, (x1_, y1_), (x2_, y2_), bgr, 2)

        for i in pik:
            if i >=y2:
                flag = "not pick up"
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.putText(img = frame,text = flag,org = (20, 20),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 1.0,color = (125, 246, 55),thickness = 3)
        cv2.line(frame, start_point, end_point, (255,0,0), 2)
        cv2.imshow("frame",frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()