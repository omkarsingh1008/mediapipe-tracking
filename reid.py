import cv2
from openvino.inference_engine import IECore
from urtils import load_reid,preprocess,draw_rec
ie = IECore()
import torch
import matplotlib.pyplot as plt

exec_net,input_layer,output_layer,size = load_reid("/media/omkar/omkar3/openvino/openvino_parallel/Openvino/person-reidentification-retail-0288/FP32/person-reidentification-retail-0288.xml")


def reidentification_P(ids,ids1,track_id,frame,frame1):
    if len(track_id)==0:
        for i,bbox in ids.items():
            img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            plt.imshow(img)
            plt.show()
            print("ffff")
            img = preprocess(img,size)
            infer_res = exec_net.start_async(request_id=0,inputs={input_layer:img})
            status=infer_res.wait()
            results = exec_net.requests[0].outputs[output_layer][0]
            for j,bbox1 in ids1.items():
                print("gggg")
                img1 = frame1[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]
                #plt.imshow(img1)
                #plt.show()
                img1 = preprocess(img1,size)
                infer_res = exec_net.start_async(request_id=1,inputs={input_layer:img1})
                status=infer_res.wait()
                results1 = exec_net.requests[1].outputs[output_layer][0]
                print(results1.shape)
                x = torch.tensor(results).unsqueeze(0)
                y = torch.tensor(results1).unsqueeze(0)
                dis=torch.cosine_similarity(x, y)[0].numpy()
                if dis >= 0.5:
                    track_id[i]= bbox


                

def reidentification(ids,track_id,frame):
    img=frame
    track_id_copy = track_id.copy()
    for i,bbox in ids.items():
        if len(track_id) == 0:
            track_id[i] = bbox
        else:
            img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            #plt.imshow(img)
            #plt.show()
            img = preprocess(img,size)
            infer_res = exec_net.start_async(request_id=0,inputs={input_layer:img})
            status=infer_res.wait()
            results = exec_net.requests[0].outputs[output_layer][0]
            dis = []
            for j,bbox_t in track_id_copy.items():
                img1 = frame[bbox_t[1]:bbox_t[3], bbox_t[0]:bbox_t[2]]
                img1 = preprocess(img1,size)
                infer_res = exec_net.start_async(request_id=1,inputs={input_layer:img1})
                status=infer_res.wait()
                results1 = exec_net.requests[1].outputs[output_layer][0]
                #print(results1.shape)
                x = torch.tensor(results).unsqueeze(0)
                y = torch.tensor(results1).unsqueeze(0)
                d=torch.cosine_similarity(x, y)[0].numpy()
                dis.append(d)
            try:
                f_idx = dis.index(max(dis))
                f = dis[f_idx]
                if f >= 0.7:
                    print("hhh")
                    track_id[f_idx] = bbox
                    i = f_idx
                else:
                    track_id[i] = bbox
                    i=i
            except Exception as e:
                print(e)
        cv2.putText(frame, str(i), bbox[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(img=frame, pt1=bbox[:2], pt2=bbox[2:], color=(0,255,0), thickness=3)
    return track_id,frame
            
            

def emb(ids,frame):
    r= []
    b=[]

    for i,bbox in ids.items():
        img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        img = preprocess(img,size)
        infer_res = exec_net.start_async(request_id=0,inputs={input_layer:img})
        status=infer_res.wait()
        results = exec_net.requests[0].outputs[output_layer][0]
        #print(type(results))
        r.append(results)
        b.append(bbox)
       
    return r,b

def ids_feature_(ids,frame):
    ids_feat={}
    for i,bbox in ids.items():
        img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        img = preprocess(img,size)
        infer_res = exec_net.start_async(request_id=0,inputs={input_layer:img})
        status=infer_res.wait()
        results = exec_net.requests[0].outputs[output_layer][0]
        ids_feat[i]=results
    return ids_feat

def distance_(feature,feat1):
    x = torch.tensor(feature).unsqueeze(0)
    y = torch.tensor(feat1).unsqueeze(0)
    d=torch.cosine_similarity(x, y)[0].numpy()
    return d


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