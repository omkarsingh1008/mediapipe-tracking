import cv2
import mediapipe as mp
import cv2
import torch
import cv2
from urtils import load,preprocess,postprocess,postprocess_yolov5
import numpy as np
import argparse
from motrackers import CentroidTracker
from urtils import load,draw_tracks,tracklets,tracking
from reid import ids_feature_,distance_,distance_list,ids_feature_list
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = model.names
model.to(device)
cap = cv2.VideoCapture("4.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width,frame_height)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output = cv2.VideoWriter('output3.mp4', fourcc, fps, frame_size)
x1,y1,x2,y2 = 0,0,0,0
tracks_id={}
tracklets_id={}

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      
      continue
    ids = {}
    tracks_draw={}
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    frame1 = [frame]
    det = model(frame1)
        
        
    bboxes, scores,labels,bboxes_bottel,scores_bottel,labels_bottel=postprocess_yolov5(x_shape, y_shape,det)
    tracks = tracker.update(bboxes, scores,labels)
        
    frame,ids = draw_tracks(frame, tracks,ids)
       
    tracks_draw,tracks_id,tracklets_id=tracking(ids,frame,tracks_id,tracklets_id,tracks_draw)
    for i,bbox in tracks_draw.items():
      image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = pose.process(image)

    
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    cv2.imshow('MediaPipe Pose', frame)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()