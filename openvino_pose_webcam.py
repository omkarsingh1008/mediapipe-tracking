import logging
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter
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
import cv2
import numpy as np
from openvino.inference_engine import IECore
import torch

import openvino_models as models
import monitors
from images_capture import open_images_capture
from pipelines import get_user_config, AsyncPipeline
from performance_metrics import PerformanceMetrics
from helpers import resolution
from urtils import load,preprocess,postprocess,postprocess_yolov5,check_id_m

default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

colors = (
        (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
        (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
        (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
        (0, 170, 255))

def draw(img,poses, point_score_threshold,output_transform,skeleton=default_skeleton,draw_ellipses=False):
    img = output_transform.resize(img)
    if poses.size == 0:
        return img
    stick_width = 4

    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:,:2].astype(np.int32)
        points = output_transform.scale(points)
        points_scores = pose[:,2]

        for i,(p,v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img,tuple(p),1,colors[i],2)
                cv2.circle(img,points[10],50,(0,0,0),-1)
                #cv2.circle(img,points[9],50,(0,0,0),-1)
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                if draw_ellipses:
                    middle = (points[i] + points[j]) // 2
                    vec = points[i] - points[j]
                    length = np.sqrt((vec * vec).sum())
                    angle = int(np.arctan2(vec[1], vec[0]) * 180 / np.pi)
                    polygon = cv2.ellipse2Poly(tuple(middle), (int(length / 2), min(int(length / 50), stick_width)),
                                               angle, 0, 360, 1)
                    cv2.fillConvexPoly(img_limbs, polygon, colors[j])
                else:
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=stick_width)
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img

def preprocess(frame,size):
    n,c,h,w = size
    input_image = cv2.resize(frame, (w,h))
    input_image = input_image.transpose((2,0,1))
    input_image.reshape((n,c,h,w))
    return input_image


#--------------------------------------------yolov5--------------------------------------------
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = model_yolov5.names
model_yolov5.to(device)    
#-----------------------------------------end--------------------------------------------------
#-----------------------------------------openvino_keypoint------------------------------------
plugin_config = {'CPU_BIND_THREAD': 'NO', 'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}
ie= IECore()
model = models.OpenPose(ie, "human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml", target_size=None, aspect_ratio=1,
                                prob_threshold=0.1)
input=model.image_blob_name
out_pool=model.pooled_heatmaps_blob_name
out_ht=model.heatmaps_blob_name
out_paf=model.pafs_blob_name
n,c,h,w = model.net.inputs[input].shape
exec_net = ie.load_network(network=model.net,config=plugin_config,device_name="CPU",num_requests = 1)
#------------------------------------------end------------------------------------------------
#-------------------------------------openvion_reid------------------------------------------
tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
exec_net_reid,input_layer_reid,output_layer_reid,size_reid = load("person-detection-retail-0013/FP32/person-detection-retail-0013.xml",num_sources=2)
tracks_id={}
tracklets_id={}
#------------------------------------------end----------------------------------------------
cap = cv2.VideoCapture("Autonomous2.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width,frame_height)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output = cv2.VideoWriter('output8.mp4', fourcc, fps, (1000,1000))
while True:
    flag = "not pick up"
    ids = {}
    id_={"":""}
    tracks_draw={}
    _,frame = cap.read()
    frame = cv2.resize(frame, (1000,1000))
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    frame1 = [frame]
    det = model_yolov5(frame1)
    #---------------------------------reid-----------------------------------
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    bboxes, scores,labels,bboxes_bottel,scores_bottel,labels_bottel=postprocess_yolov5(x_shape, y_shape,det)
    tracks = tracker.update(bboxes, scores,labels)
        
    frame,ids = draw_tracks(frame, tracks,ids)
       
    tracks_draw,tracks_id,tracklets_id=tracking(ids,frame,tracks_id,tracklets_id,tracks_draw)
    #---------------------------------keypoint--------------------------------
    output_transform = models.OutputTransform(frame.shape[:2], None)
    output_resolution = (frame.shape[1], frame.shape[0])
    inputs, preprocessing_meta = model.preprocess(frame)
    infer_res = exec_net.start_async(request_id=0,inputs={input:inputs["data"]})
    status=infer_res.wait()
    results_pool = exec_net.requests[0].outputs[out_pool]
    results_ht = exec_net.requests[0].outputs[out_ht]
    results_paf = exec_net.requests[0].outputs[out_paf]
    results={"heatmaps":results_ht,"pafs":results_paf,"pooled_heatmaps":results_pool}
    poses,scores=model.postprocess(results,preprocessing_meta)
    #print(poses)
    #----------------------------------------------------------------------------
    #------------------------------bottle----------------------------------------
    labels, cord = det.xyxyn[0][:, -1], det.xyxyn[0][:, :-1]
    
    X = int(x_shape * (20/100))
    start_point = (X,0)
    end_point = (X,y_shape)
    pik=[]
    bol=[]
    for i in range(len(cord)):
        bbox=cord[i]
        c=labels[i]
           
        if int(c)==39:
            x1_, y1_, x2_, y2_ = int(bbox[0] * x_shape), int(bbox[1] * y_shape), int(bbox[2] * x_shape), int(bbox[3] * y_shape)
                
            bgr = (0, 255, 0)
            #pik.append(x1_)
            #cv2.circle(frame,(x2_,y1_),50,(0,0,0),-1)
            pik.append([[x1_, y1_, x2_, y2_],x1_])
            if x1_>=X:
                bol.append([x1_, y1_, x2_, y2_])
            center_right = (int((x2_+x2_)/2),int((y2_+y1_)/2))
            cv2.circle(frame,center_right,50,(0,0,0),1)
            cv2.rectangle(frame, (x1_, y1_), (x2_, y2_), bgr, 2)
    #points = poses[:,:2].astype(np.int32)
    points = output_transform.scale(poses)
    #print((points[0]))
    
    for i in pik:
        if i[-1] >=X:
            flag = "pick up"
    id_ = check_id_m(bol,points,tracks_draw)
            
    #----------------------------------------------------------------------------
    print(id_)
    font = cv2.FONT_HERSHEY_SIMPLEX
    d=0
    for id,f in id_.items():
            cv2.putText(frame, str(id)+f, (10, 30+d), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            d+=30
    for id,bbox in tracks_draw.items():
        cv2.putText(frame, str(id), bbox[:2], 1, cv2.FONT_HERSHEY_DUPLEX, (0, 0, 255), 3)
        cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 255, 0), 1)
    frame = draw(frame,poses,0.1,output_transform)
    cv2.putText(img = frame,text = flag,org = (20, 20),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 1.0,color = (125, 246, 55),thickness = 3)
    cv2.line(frame, start_point, end_point, (255,0,0), 2)
    output.write(frame)
    cv2.imshow('smart store', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
output.release()
cv2.destroyAllWindows()