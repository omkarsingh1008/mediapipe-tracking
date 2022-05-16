import cv2
from urtils import load,preprocess,postprocess
import numpy as np
import argparse
from motrackers import CentroidTracker
from urtils import load,draw_tracks,tracklets,tracking
from reid import ids_feature_,distance_,distance_list,ids_feature_list
import time
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = model.names
model.to(device)

def main(filename_path,source):
 
    exec_net,input_layer,output_layer,size = load(filename_path,num_sources=2)
    tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
    tracker1 = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
    vid = cv2.VideoCapture(int(source[0]))
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    frame_size = (frame_width,frame_height)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    output = cv2.VideoWriter('output2.mp4', fourcc, fps, frame_size)
    
    tracks_id={}
    tracklets_id={}
    prev_frame_time = 0
    new_frame_time = 0
    x1,y1,x2,y2 = 0,0,0,0
    while(True):
        flag = "pick up"
        ids = {}
        ids1={}
        tracks_draw={}
        
        ret, frame = vid.read()
        
        input_image = preprocess(frame,size)
        infer_res = exec_net.start_async(request_id=0,inputs={input_layer:input_image})
    
        status=infer_res.wait()
        results = exec_net.requests[0].outputs[output_layer][0][0]
        
        bboxes, scores,labels,frame = postprocess(frame,results)
        
        
        tracks = tracker.update(bboxes, scores,labels)
        
        frame,ids = draw_tracks(frame, tracks,ids)
       
        tracks_draw,tracks_id,tracklets_id=tracking(ids,frame,tracks_id,tracklets_id,tracks_draw)
        

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
            if int(c)==39:
                x1_, y1_, x2_, y2_ = int(bbox[0] * x_shape), int(bbox[1] * y_shape), int(bbox[2] * x_shape), int(bbox[3] * y_shape)
                bgr = (0, 255, 0)
                pik.append(y2_)
                cv2.rectangle(frame, (x1_, y1_), (x2_, y2_), bgr, 2)

        for i in pik:
            if i >=y2:
                flag = "not pick up"
            
        for id,bbox in tracks_draw.items():
            cv2.putText(frame, str(id), bbox[:2], 1, cv2.FONT_HERSHEY_DUPLEX, (0, 0, 255), 3)
            cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 255, 0), 1)
       

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        
        output.write(frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img = frame,text = flag,org = (20, 20),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 1.0,color = (125, 246, 55),thickness = 3)
        cv2.line(frame, start_point, end_point, (255,0,0), 2)
        cv2.putText(frame, str(fps), (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        output.write(frame)
        cv2.imshow('Multi camera tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    vid.release()

    output.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--weight',default="")
    args.add_argument('-s', '--source', required=True, nargs='+',
                        help='Input sources (indexes of cameras or paths to video files)')
    parsed_args=args.parse_args()
    main(filename_path=parsed_args.weight,source=parsed_args.source)