# Studying-Machine-Deep-Learning
Summarize and storage of machine learning and deep learning, for personal study purpose

My complete items are as below:
1. Tensorflow X Keras_Building Blocks.ipynb
2. Tensorflow_Building Blocks.ipynb
3. 

Below are the items have been studied and to keep as reference in the future:
1. **Convolution model-Application-v1.ipynb**  *from Andrew Ng Deep Learning Specialization Course 4 Week 1*
	- Introduce the building process of ConvNet in Tensorflow, with minibatch training and randomize minibatch per epoch
2. **Art Generation with Neural Style Transfer-v2.ipynb**  *from Andrew Ng Deep Learning Specialization Course 4 Week 4*
	- Define pixels of target picture as trainable variables, and use the 'difference' of pictures loss function
	- Pretrained model does not include in repo, and can be downloaded from [this link ](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)
3. **Autonomous driving application-Car detection-v1.ipynb** *from Andrew Ng Deep Learning Specialization Course 4 Week 3* (NOT COLLECTED IN REPO, [Original Source](https://www.coursera.org/learn/convolutional-neural-networks/notebook/bbBOL/car-detection-with-yolov2))
	- Create prob filter and non-max suppression for YOLO model
	- If the input image is not the same size as YOLO model
		1. rescale the image to fit YOLO model
		2. forward propagation to output
		3. prob filter (score-thresholding)
		4. rescale the bounding boxes back to original scale --> so that IoU can be correctly evaluated
		5. non-max suppression
