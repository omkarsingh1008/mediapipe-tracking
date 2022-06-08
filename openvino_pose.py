import logging
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from openvino.inference_engine import IECore

import models
import monitors
from images_capture import open_images_capture
from pipelines import get_user_config, AsyncPipeline
from performance_metrics import PerformanceMetrics
from helpers import resolution

default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

colors = (
        (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
        (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
        (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
        (0, 170, 255))
def preprocess(frame,size):
    n,c,h,w = size
    input_image = cv2.resize(frame, (w,h))
    input_image = input_image.transpose((2,0,1))
    input_image.reshape((n,c,h,w))
    return input_image
ie= IECore()
img1 = cv2.imread("1648123042.jpg")
model = models.OpenPose(ie, "human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml", target_size=None, aspect_ratio=1,
                                prob_threshold=0.1)
plugin_config = {'CPU_BIND_THREAD': 'NO', 'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}
hpe_pipeline = AsyncPipeline(ie, model, plugin_config, device="CPU", max_num_requests=0)

inputs, preprocessing_meta = model.preprocess(img1)
print(inputs["data"])
input=model.image_blob_name
out_pool=model.pooled_heatmaps_blob_name
out_ht=model.heatmaps_blob_name
out_paf=model.pafs_blob_name



#input_layer = next(iter(model.net.inputs))
n,c,h,w = model.net.inputs[input].shape
exec_net = ie.load_network(network=model.net,config=plugin_config,device_name="CPU",num_requests = 1)
requests=exec_net.requests








for _ in range(5):
    output_layer = next(iter(model.net.outputs))
    print(output_layer)
input_image = preprocess(img1,(n,c,h,w))
infer_res = exec_net.start_async(request_id=0,inputs={input:inputs["data"]})
status=infer_res.wait()
results_pool = exec_net.requests[0].outputs[out_pool]
results_ht = exec_net.requests[0].outputs[out_ht]
results_paf = exec_net.requests[0].outputs[out_paf]
results={"heatmaps":results_ht,"pafs":results_paf,"pooled_heatmaps":results_pool}
poses,scores=model.postprocess(results,preprocessing_meta)

print(results_pool.shape)
print(results_ht.shape)
print(results_paf.shape)

print(poses)
print(scores)
print("*"*50)
print(input)
print("*"*50)
print((n,c,h,w))
print("*"*50)
print(out_pool)
output_transform = models.OutputTransform(img1.shape[:2], None)
output_resolution = (img1.shape[1], img1.shape[0])
#print(len(results))

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


img = draw(img1,poses,0.1,output_transform)

cv2.imshow('image',img)
cv2.waitKey(0)