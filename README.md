# Machine Learning and Deep Learning
内容以时间倒序（由近期到早期）排列。

## 学习课程

1. [Udacity Data Analyst (Advanced)](https://cn.udacity.com/course/data-analyst-nanodegree--nd002-cn-advanced) --> 进行中 (2018)
2. [Deep Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/deep-learning) --> [完成证书](https://www.coursera.org/account/accomplishments/specialization/certificate/MAJJ6QCYCYTX)  (2017-2018)
3. [Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning/) --> [完成证书](https://www.coursera.org/account/accomplishments/certificate/A4DF5DYNZENU) (2018)
4. [Microsoft Professional Program in Data Science](https://www.edx.org/microsoft-professional-program-data-science) --> [完成证书](https://academy.microsoft.com/en-us/certificates/7539ddd1-5a3a-4bfe-9c0b-a2ed2bb42b8f/) (2016-2017) 

## 项目列表

### 循环神经网络 RNN



### 卷积神经网络 CNN

1. [ResNet-50：数字手势识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/ResNets/Residual%20Networks%20-%20v2.ipynb) （Identity / Convolutional Block；用Keras构建并训练模型；未上传pre-trained model）
2. [CNN：笑脸识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Keras%20for%20Happy%20Face/Keras%20-%20Tutorial%20-%20Happy%20House%20v2.ipynb) （ZeroPad + Conv2D + BatchNorm + ReLu + MaxPool；用Keras构建并训练模型）
3. [CNN：数字手势识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/CNN%20for%20Signs/Convolution%20model%20-%20Application%20-%20v1.ipynb) （[Conv2D + ReLu + MaxPool]*2；用Tensorflow构建并训练模型）

### 深度神经网络 DNN

1. [三隐藏层：数字手势识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Tensorflow%20for%20Signs/Tensorflow%20Tutorial.ipynb) （用Tensorflow构建并训练模型）
2. [多隐藏层：分辨图片是否为猫](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Deep%20Neural%20Network%20Application_%20Image%20Classification/Deep%20Neural%20Network%20-%20Application%20v3.ipynb) （研究隐藏层数对于结果的影响；用Numpy构建并训练模型）
3. [单隐藏层：分辨二元图形](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Planar%20data%20classification%20with%20one%20hidden%20layer/Planar%20data%20classification%20with%20one%20hidden%20layer%20v4.ipynb) （研究隐藏单元数量对于结果的影响；用Numpy构建并训练模型）
   - ![](img\single_layer_NN.png)
4. [以梯度下降训练逻辑回归模型：分辨图片是否为猫](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Logistic%20Regression%20as%20a%20Neural%20Network/Logistic%20Regression%20with%20a%20Neural%20Network%20mindset%20v4.ipynb) （预处理图片文件，用Numpy定义正反向传播算法、sigmoid激活函数和成本函数，训练模型并进行预测）

##心得和总结

### 循环神经网络 RNN



### 卷积神经网络 CNN

1. [用Numpy构建卷积神经网络正向传播算法](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/CNN%20for%20Signs/Convolution%20model%20-%20Step%20by%20Step%20-%20v2.ipynb) （Zero-padding, Convolution, Max / Average Pooling）

### 深度神经网络 DNN

1. [Keras使用指南及模型可视化](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Tensorflow_X_Keras_Building%20Blocks.ipynb) 
2. [Tensorflow使用指南及TensorBoard调用](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Tensorflow_Building_Blocks.ipynb) 
3. [不同梯度下降方法对深度神经网络的影响]() （Stochastic / Mini-batch gradient descent, Momentum, Adam）
4. [通过梯度检查确认算法正确](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Gradient%20Checking/Gradient%20Checking%20v1.ipynb) （Gradient directly estimated by cost function should be same as calculated by formulas）
5. [不同正则化方法对深度神经网络的影响](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Regularization/Regularization.ipynb) （None / L2 / Dropout）
6. [不同初始化方法对深度神经网络的影响](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Initialization/Initialization.ipynb) （Zero / Random / He Initialization）
7. [用Numpy构建多隐藏层深度神经网络](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step%20v5.ipynb) 
8. [深度学习中的Numpy使用](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Python_Basics_with_Numpy/Python%20Basics%20With%20Numpy%20v3.ipynb) (Reshape, Normalization, Broadcasting, Vectorization)

### 机器学习

1. [决策树算法理论](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Machine%20Learning/Tree_Based_Algorithm_Related_Topics.ipynb)（随机森林原理，Bootstrap采样，OOB误差评估）

### Python基础

1. [Class面向对象编程](https://raw.githubusercontent.com/YestinYang/Studying-Machine-Deep-Learning/master/Basic%20Python/Class_OOP.ipynb) (Class / Instance Variable, Regular / Class / Static Method, Inheritance, Dunder Method, Decorators)
2. Python编程小实验
  - 创建网站介绍喜欢的电影 --> 进行中
  - [`turtle` 绘图](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/drawing_turtle.py) / [工作间隔休息提醒器](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/take_break.py) / [用词不当检测器](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/word_checker.py) / [批量文件自定义规则重命名](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/rename.py)


### 其他

1. [从Jupyter Notebook平台中打包并下载所有文件](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Download_Files_Jupyter_Hub.ipynb) 
2. [让Jupyter Notebook显示同个cell的多个outputs](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Anaconda%20Related/ipython_config.py) 

------

Summarize and storage of machine learning and deep learning, for personal study purpose

My complete items are as below:

1. Tensorflow X Keras_Building Blocks.ipynb
2. Tensorflow_Building Blocks.ipynb
3. ​

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
