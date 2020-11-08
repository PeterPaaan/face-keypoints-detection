# face-keypoints-detection
Detection 15 key points on faces and mask a specific area with a token. Model based on `OpenCV` and `Keras`.

## How to play with
<<<<<<< HEAD
To have a quick glance on how this project works, simply run `face.keypoints` by loading the pre_trained model `demo_pretrained_model.h5`. The camera on your laptop will be activated and start to detect faces. A pair of sunglasses will add to the detected face.
  
||||||| merged common ancestors
To have a quick glance on how this project works, simply run `face.keypoints` by loading the pre_trained model `demo_pretrained_model.h5`.  
=======
To have a quick glance on how this project works, simply run `face.keypoints.py` by loading the pre_trained model `demo_pretrained_model.h5`.The camera on your laptop will be activated and start to detect faces. A pair of sunglasses will add to the detected face.  
>>>>>>> 57a8df749c9d8c5c13ecc479aca00d9cbff6e195
The detection performance looks like below:  
  
![face](model_performance.png)  

## Train your own model
The model structre in `kmodel` follows a classical convolutional layers + FC layers here, you can try and make changes to get better performace. (First get your training data, which can be found on kaggle.)
