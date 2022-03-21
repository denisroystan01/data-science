#!/usr/bin/env python
# coding: utf-8

# ## Capstone Project - Driver Drowsiness Detection 

# > Python version 3.7.3 | Tensorflow 2 Object Detection Model

# ### Imports

# In[2]:


import numpy as np
import tensorflow as tf

import time
from playsound import playsound

# Import the object detection module.

# In[3]:


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# Patches:

# In[4]:


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


# # Model preparation 

# In[5]:


def load_model():
    model_dir = "saved_model"
    model = tf.saved_model.load(str(model_dir))
    return model


# ## Loading label map

# In[6]:


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# # Detection

# Load an object detection model:

# In[8]:


detection_model = load_model()


# Add a wrapper function to call the model, and cleanup the outputs:

# In[9]:


def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


# ### Alarm Code

# In[ ]:


counter = 0


# In[11]:


def alarm_code(class_id):
    global counter
    
    if int(class_id) == 1:
        counter = counter + 1
    else:
        counter = 0
    
    if counter == 5:
        counter = 0
        playsound('alarm.mp3', block = False)


# ### Camera Detection and Inference

# In[12]:


import cv2
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, image_np = cap.read()
    # Actual detection.
    output_dict = run_inference_for_single_image(detection_model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=4)
    alarm_code(str(output_dict['detection_classes'][0]))
    cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    time.sleep(1)

cap.release()
cv2.destroyAllWindows()

