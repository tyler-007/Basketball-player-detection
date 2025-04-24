1. Dataset Selection & Preprocessing: Big_Vision_assignment_preprocessing.ipynb
○ Use any open-source basketball gameplay video dataset.
○ Ensure high-quality frame extraction and annotation.
○ Preprocess videos (e.g., resizing, normalization, frame sampling).

– https://codalab.lisn.upsaclay.fr/competitions/12424
–https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdGplTHE3WW5ZR1JnUVBtdUFrd1dobmRPYzQxP2U9SXRKYWFh&cid=91819DD8AE2EDED8&id=91819DD8AE2EDED8%21131&parId=91819DD8AE2EDED8%21130&o=OneUp

This dataset had Train, Test and Val with a gt.txt file for each folder. Moreover, this dataset had football, volleyball and basketball. 

So, the basketball dataset was separated into a different dataset (splitting.py) and since the dataset had already been annotated but in MOT, the gt.txt was converted for YOLO.txt files (mot_to_yoloTXT.py) 

Now, train and val had 16-17 folders each containing images in the format 00001.jpg 00002.jpg …… and also for each sub folder in train and val train_labels and val_labels also had those 16-17 folder with with txt files like 00001.txt 00002.txt etc.

2. Model Development: - 
○ Train a detection model (e.g., YOLO, Faster R-CNN, DETR) to detect players on the court.
○ Implement a tracking algorithm (e.g., SORT, DeepSORT, ByteTrack) to follow players throughout the     video.
○ Optimize performance for real-time or near real-time tracking.

To make it compatible with YOLO training it was flattened with {folder_name}+{image_offset}.jpg and similarly for txt and two folders train and val were obtained (flattening.py)

Data.yaml file used for the YOLO model. The model was trained for 10 Epochs, images = 640 (all images were of size 1280*720) batch 16 and rect= true (for better scaling)

Big_Vision_Assignment.ipynb

After 10 epochs - 
 And also on val 


To test the model detection - the detection folder has also been uploaded

Train4 best.pt was chosen as the final model. The final results are available - train4

The model was also tested for detection on real time and also on videos - videos are available here - Output on videos


For tracking ByteTrack was used for giving each player a unique id. We generated tracking data as tracker_output.txt (tracker_output.txt) and for real time tracking - Big_vision_assignment_tracking.ipynb

The above file also includes metrics calculation like Results for IoU=0.7, Conf=0.4 -> MOTA: 0.952755905511811, IDF1: 0.6722644037516748, MOTP: 0.09114055193985346, the tracking video as also available on - Moreover, for trying to increase the MOTP - Kalman Filter was also applied.
testing metrics


