import cv2
import mediapipe as mp
import torch
import cv2
from urtils import load,preprocess,postprocess,postprocess_yolov5
import numpy as np
import argparse
from motrackers import CentroidTracker
from urtils import load,draw_tracks,tracklets,tracking,check_id
from reid import ids_feature_,distance_,distance_list,ids_feature_list
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
exec_net,input_layer,output_layer,size = load("person-detection-retail-0013/FP32/person-detection-retail-0013.xml",num_sources=2)
tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
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
output = cv2.VideoWriter('output7.mp4', fourcc, fps, frame_size)
x1,y1,x2,y2 = 0,0,0,0
tracks_id={}
tracklets_id={}
draw_p={"draw":None}

prev_frame_time = 0
new_frame_time = 0
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while True:
        flag = "not pick up"
        ids = {}
        tracks_draw={}
        id_={"":""}
        _,frame = cap.read()

        x_shape, y_shape = frame.shape[1], frame.shape[0]
        frame1 = [frame]
        det = model(frame1)
        
        
        bboxes, scores,labels,bboxes_bottel,scores_bottel,labels_bottel=postprocess_yolov5(x_shape, y_shape,det)
        
        
        tracks = tracker.update(bboxes, scores,labels)
        
        frame,ids = draw_tracks(frame, tracks,ids)
       
        tracks_draw,tracks_id,tracklets_id=tracking(ids,frame,tracks_id,tracklets_id,tracks_draw)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = pose.process(frame)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        frame1 = [frame]
        det = model(frame1)
    
        labels, cord = det.xyxyn[0][:, -1], det.xyxyn[0][:, :-1]
    
        X = int(x_shape * (50/100))
        start_point = (X,0)
        end_point = (X,y_shape)
        pik=[]
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        lmList = []
        if results.pose_landmarks:
            mypose = results.pose_landmarks
            for id, lm in enumerate(mypose.landmark):
                
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                lmList.append([id, cx, cy])
        for i in range(len(cord)):
            bbox=cord[i]
            c=labels[i]
           
            if int(c)==39:
                x1_, y1_, x2_, y2_ = int(bbox[0] * x_shape), int(bbox[1] * y_shape), int(bbox[2] * x_shape), int(bbox[3] * y_shape)
                
                bgr = (0, 255, 0)
                pik.append([[x1_, y1_, x2_, y2_],x1_])
                cv2.rectangle(frame, (x1_, y1_), (x2_, y2_), bgr, 2)

        for i in pik:
            if i[-1] >=X:
                flag = "pick up"
                id_ = check_id(i,lmList,tracks_draw)

        
        if  flag == "pick up":
            for id,bbox in tracks_draw.items():
                draw_p["draw"]=str(id)+":-"+"pick up"
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            draw_p["draw"]=None
            cv2.putText(frame, "not pick up", (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        for id,bbox in tracks_draw.items():
            cv2.putText(frame, str(id), bbox[:2], 1, cv2.FONT_HERSHEY_DUPLEX, (0, 0, 255), 3)
            cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 255, 0), 1)
       

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        d=0
        print(id_)
        for id,f in id_.items():
            cv2.putText(frame, str(id)+f, (10, 30+d), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            d+=30
        cv2.line(frame, start_point, end_point, (255,0,0), 2)
        #cv2.putText(frame, str(fps), (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        output.write(frame)
        frame = cv2.resize(frame,(800,800))
        cv2.imshow("frame",frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    output.release()
    cv2.destroyAllWindows()
