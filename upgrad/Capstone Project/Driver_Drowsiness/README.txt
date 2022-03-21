### Capstone Project - Driver Drowsiness Detection

##### Python Version 3.7.3

#### The Detection Model to detect whether the eyes of a person is sleepy or awake. If the user is sleepy for more than 5 secs than the alarm will ring up.


> Label Map
	- Sleepy (eye_closed)
	- Awake (eye_opened)

> Libraries required to run the python file
	- opencv-python [pip install opencv-python] (To run the camera and capture the frames)
	- pycocotools [pip install pycocotools]
	- Tensorflow 2.x [pip install tensorflow] (Make sure tensorflow installed version is 2.x)
	- tf_slim [pip install tf_slim]
	- playsound [pip install playsound] (Library for running alarm sound)
	- Tensorflow Object Detection API


> Tensorflow Pre-trained Model Used;
	- SSD MobileNet V2 FPNLite 320x320 (http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz)
	- COCO mAP [22.2]
	- Speed (ms) [22]

> Model Pipeline Config Changes:
	- Changed the "num_classes" to 2 as we want to detect two custom objects (Sleepy or Awake)
	- Changed the "batch_size" to 24
	- Passed the paths to .record (train and test) files and .pbtxt (label_map) file
	- Changed the path of "fine_tune_checkpoint" to the used pre-trained model checkpoint
	- Changed the "fine_tune_checkpoint_type" to "detection" for detecting custom objects

> Files
	- Frozen_inference_graph (saved_model/saved_model.pb)
	- Label map file (label_map.pbtxt)
	- Model config file (model.config)
	- .record files (test.record & train.record)
	- Alarm file (alarm.mp3)
	- Final Python file to run the detection (driver_drowsiness_with_alarm.py)

> To run the final python file
	- Make sure the above mentioned libraries are installed
	- Open the Terminal in the same directory where the particular file "driver_drowsiness_with_alarm.py" resides and run the below command
		> [python driver_drowsiness_with_alarm.py]