import cv2
import numpy as np
from openvino.inference_engine import IECore
#from reid import ids_feature_,distance_,distance_list,ids_feature_list
import torch

ie = IECore()
def load(filename,num_sources = 1):
    filename_bin = filename.split('.')[0]+".bin"
    net = ie.read_network(model = filename,weights = filename_bin)
    input_layer = next(iter(net.inputs))
    n,c,h,w = net.inputs[input_layer].shape
    exec_net = ie.load_network(network=net,device_name="CPU",num_requests = num_sources)
    output_layer = next(iter(net.outputs))

    return exec_net,input_layer,output_layer,(n,c,h,w)

def preprocess(frame,size):
    n,c,h,w = size
    try:
        input_image = cv2.resize(frame, (w,h))
    except:
        input_image = np.zeros((512,512,3))
        input_image = cv2.resize(input_image, (w,h))
    input_image = input_image.transpose((2,0,1))
    input_image.reshape((n,c,h,w))
    return input_image

def postprocess(frame,results):
    boxes = []
    labels = []
    scores = []
    for _, label, score, xmin, ymin, xmax, ymax in results:
        h1, w1 = frame.shape[:2]
        
        boxes.append(list(map(int, (xmin * w1, ymin * h1, (xmax - xmin)*w1, (ymax - ymin) * h1))))
        labels.append(int(label))
        scores.append(float(score))
    indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=0.6, nms_threshold=0.7)
    

    boxes=[(boxes[idx]) for idx in indices]
    labels=[labels[idx] for idx in indices]
    scores=[scores[idx] for idx in indices]
    #for  box in boxes:
    
    #    cv2.rectangle(img=frame, pt1=box[:2], pt2=box[2:], color=(0,255,0), thickness=3)

    
    return boxes,scores,labels,frame
    
def xywh2xyxy(xywh):

    if len(xywh.shape) == 2:
        x = xywh[:, 0] + xywh[:, 2]
        y = xywh[:, 1] + xywh[:, 3]
        xyxy = np.concatenate((xywh[:, 0:2], x[:, None], y[:, None]), axis=1).astype('int')
        return xyxy
    if len(xywh.shape) == 1:
        x, y, w, h = xywh
        xr = x + w
        yb = y + h
        return np.array([x, y, xr, yb]).astype('int')

def draw_tracks(image, tracks,ids):
    for trk in tracks:

        trk_id = trk[1]
        xmin = trk[2]
        ymin = trk[3]
        width = trk[4]
        height = trk[5]
        
        #xmax = xmin + width
        #ymax = ymin + height
        b=np.array([xmin,ymin, width, height])
        bbox=xywh2xyxy(b)

        xcentroid, ycentroid = int(xmin + 0.5*width), int(ymin + 0.5*height)

        text = "ID {}".format(trk_id)

        cv2.putText(image, text, (xcentroid - 10, ycentroid - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(image, (xcentroid, ycentroid), 4, (0, 255, 0), -1)
        cv2.rectangle(img=image, pt1=bbox[:2], pt2=bbox[2:], color=(0,255,0), thickness=3)
        ids[trk_id]=bbox


        

    return image,ids

def load_reid(filename,num_sources = 2):
    filename_bin = filename.split('.')[0]+".bin"
    net = ie.read_network(model = filename,weights = filename_bin)
    input_layer = next(iter(net.inputs))
    n,c,h,w = net.inputs[input_layer].shape
    exec_net = ie.load_network(network=net,device_name="CPU",num_requests = num_sources)
    output_layer = next(iter(net.outputs))

    return exec_net,input_layer,output_layer,(n,c,h,w)


exec_net,input_layer,output_layer,size = load_reid("/media/omkar/omkar3/openvino/openvino_parallel/Openvino/person-reidentification-retail-0288/FP32/person-reidentification-retail-0288.xml")

def draw_rec(frame,track_id):
    for i,bbox in track_id.items():
        cv2.putText(frame, str(i), bbox[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(img=frame, pt1=bbox[:2], pt2=bbox[2:], color=(0,255,0), thickness=3)
    return frame

def distance_list(feature,feat1):
    x = torch.tensor(feature[0]).unsqueeze(0)
    dis=[]
    for i in feat1:
        y = torch.tensor(i).unsqueeze(0)
        d=torch.cosine_similarity(x, y)[0].numpy()
        dis.append(d)
    
    return sum(dis)/len(dis)

def ids_feature_list(ids,frame):
    ids_feat={}
    for i,bbox in ids.items():
        img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        img = preprocess(img,size)
        infer_res = exec_net.start_async(request_id=0,inputs={input_layer:img})
        status=infer_res.wait()
        results = exec_net.requests[0].outputs[output_layer][0]
        ids_feat[i]=[results]
    return ids_feat


def tracklets(max_key,feature,track_id,tracklets_id):
    if max_key not in tracklets_id:
        tracklets_id[max_key]=0
    tracklets_id[max_key] = tracklets_id[max_key]+1

    track_id[max_key].insert(tracklets_id[max_key],feature[0])

    if tracklets_id[max_key] > 50:
        tracklets_id[max_key] = 0

    return tracklets_id,track_id


                

def tracking(ids,frame,tracks_id,tracklets_id,tracks_draw):
    ids_feat = ids_feature_list(ids,frame)

    if len(tracks_id)==0:
        tracks_id = ids_feat
        tracks_draw = ids
        for i in ids:
            tracklets_id[i] = 0
    else:
        for i,feature in ids_feat.items():
            dis={}
                #print(type(feature))
            for id,feat1 in tracks_id.items():
                    #print(len(feat1))
                d = distance_list(feature,feat1)
                    #print(d)
                dis[id]=d
            max_key = max(dis, key=dis.get)
            #print("first_id:-",i)
            #print("track_id:-",id)
            print("reid:-",max_key)
            #print("reid_dis:-",dis[max_key])
           # print("reid_dis:-",dis)
                
            if dis[max_key] > .5:
                    #tracks_id[max_key] = feature
                tracklets_id,tracks_id = tracklets(max_key,feature,tracks_id,tracklets_id)
                    #if max_key not in tracklets_id:
                   #     tracklets_id[max_key]=0
                   # tracklets_id[max_key] = tracklets_id[max_key]+1
                tracks_draw[max_key] = ids[i]
            else:
                tracks_id[len(tracks_id)+1] = feature
                tracks_draw[len(tracks_id)+1] = ids[i]
    return tracks_draw,tracks_id,tracklets_id


def postprocess_yolov5(x_shape, y_shape,results):
    boxes_person = []
    labels_person = []
    scores_person = []
    boxes_bottel = []
    labels_bottel = []
    scores_bottel = []
    labels,scores, cord = results.xyxyn[0][:, -1],results.xyxyn[0][:, -2], results.xyxyn[0][:, :-2]
    for i,l in enumerate(labels):
        if int(l) == 0:
            bbox = cord[i]
            boxes_person.append([int(bbox[0] * x_shape), int(bbox[1] * y_shape), int((bbox[2]-bbox[0]) * x_shape), int((bbox[3]-bbox[1]) * y_shape)])
            labels_person.append(int(labels[i]))
            scores_person.append(float(scores[i]))
        if int(l) == 39:
            bbox = cord[i]
            boxes_bottel.append([int(bbox[0] * x_shape), int(bbox[1] * y_shape), int((bbox[2]-bbox[0]) * x_shape), int((bbox[3]-bbox[1]) * y_shape)])
            labels_bottel.append(int(labels[i]))
            scores_bottel.append(float(scores[i]))
    return boxes_person,scores_person,labels_person,boxes_bottel,scores_bottel,labels_bottel

    
    