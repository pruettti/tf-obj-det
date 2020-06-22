#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import json
import tempfile
import logging
import shutil
import math

import time

MODEL_BASE = '/opt/models/research'
sys.path.append(MODEL_BASE)
sys.path.append(MODEL_BASE + '/object_detection')
sys.path.append(MODEL_BASE + '/slim')

from flask import Flask, request
import numpy as np
import requests
from PIL import Image
import tensorflow as tf
from utils import label_map_util

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

app.logger.info(sys.version)


PATH_TO_CKPT = '/opt/graph_def/frozen_inference_graph.pb'
PATH_TO_LABELS = '/opt/tf-obj-det/labelmap.pbtxt'

content_types = {'jpg': 'image/jpeg',
                 'jpeg': 'image/jpeg',
                 'png': 'image/png'}
extensions = sorted(content_types.keys())

class ObjectDetector(object):

  def __init__(self):
    self.detection_graph = self._build_graph()
    self.sess = tf.Session(graph=self.detection_graph)

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    category_codes = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=False)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)
    self.category_code_index = label_map_util.create_category_index(category_codes)

  def _build_graph(self):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    return detection_graph

  def _load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

  def detect(self, image):
    image_np = self._load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    graph = self.detection_graph
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    boxes = graph.get_tensor_by_name('detection_boxes:0')
    scores = graph.get_tensor_by_name('detection_scores:0')
    classes = graph.get_tensor_by_name('detection_classes:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = self.sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    boxes, scores, classes, num_detections = map(
        np.squeeze, [boxes, scores, classes, num_detections])

    return boxes, scores, classes.astype(int), num_detections

def detect_objects(image_path, bucket_name, source_blob_name):
  image = Image.open(image_path).convert('RGB')
  im_width, im_height = image.size

  ts = time.time()
  print('Started detection at: {}'.format(ts))
  boxes, scores, classes, num_detections = client.detect(image)
  ts2 = time.time()
  print('Ended detection at:   {}'.format(ts2))
  print('Detection duration:   {}'.format(ts2 - ts))
  sys.stdout.flush()

  detections = []
  for i in range(num_detections):
    if scores[i] < 0.7: continue
    ymin, xmin, ymax, xmax = boxes[i]
    det_class = classes[i]
    det_score = scores[i]
    detections.append(dict({'box': {'ymin': int(math.floor(ymin * im_height)), 'xmin': int(math.floor(xmin * im_width)), 'ymax': int(math.ceil(ymax * im_height)), 'xmax': int(math.ceil(xmax * im_width))}, 'score': float(det_score), 'class': client.category_code_index[det_class]['name']}))

  meta = dict({'start_timestamp': ts, 'end_timestamp': ts2, 'inference_duration': (ts2 - ts), 'im_width': im_width, 'im_height': im_height, 'bucket_name': bucket_name, 'source_blob_name': source_blob_name})
  result_json = json.dumps(dict({'detected_objs': detections, 'meta': meta}))

  app.logger.info(result_json)

  return result_json


@app.route('/detect', methods=['GET', 'POST'])
def process_image():
  if request.json:
    data = request.json

    bucket_name = data.get('bucket_name')
    source_blob_name = data.get('source_blob_name')
    temp_file = '/tmp/temp.jpg'

    app.logger.info(bucket_name)
    app.logger.info(source_blob_name)

#    download_blob(bucket_name, source_blob_name, temp_file)

    result = detect_objects(temp_file, bucket_name, source_blob_name)

    return result

client = ObjectDetector()

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80, debug=False)