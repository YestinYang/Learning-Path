# Machine Learning and Deep Learning
内容以时间倒序（由近期到早期）排列。

## 学习清单

1. [Udacity Data Analyst (Advanced)](https://cn.udacity.com/course/data-analyst-nanodegree--nd002-cn-advanced) --> 进行中 (2018)
2. [Deep Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/deep-learning) --> [完成证书](https://www.coursera.org/account/accomplishments/specialization/certificate/MAJJ6QCYCYTX)  (2017-2018)
3. [Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning/) --> [完成证书](https://www.coursera.org/account/accomplishments/certificate/A4DF5DYNZENU) (2018)
4. [Microsoft Professional Program in Data Science](https://www.edx.org/microsoft-professional-program-data-science) --> [完成证书](https://academy.microsoft.com/en-us/certificates/7539ddd1-5a3a-4bfe-9c0b-a2ed2bb42b8f/) (2016-2017) 

------

## 项目列表

### 循环神经网络 RNN



### 卷积神经网络 CNN

1. [VGG-19迁移学习：艺术风格化照片](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Neural%20Style%20Transfer/Art%20Generation%20with%20Neural%20Style%20Transfer%20-%20v2.ipynb) （in Tensorflow；将绘画作品的艺术风格迁移到日常照片中，合成新图片；成本函数 = 新旧照片的输出差异程度 + 新旧照片在数个选定卷积层的输出的格拉姆矩阵差异程度；[莫奈《亚嘉杜的罂粟花田》+ 法国罗浮宫训练过程图示](https://raw.githubusercontent.com/YestinYang/Studying-Machine-Deep-Learning/master/img/Art_Transfer_Procedure.png)；未上传pre-trained VGG-19）
2. [FaceNet：人脸识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Face%20Recognition/Face%20Recognition%20for%20the%20Happy%20House%20-%20v3.ipynb) （in Keras；1对N人脸匹配问题；Triplet loss function, L2 distance；未上传pre-trained FaceNet）
3. [YOLO：自动驾驶中的车辆识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Car%20detection%20for%20Autonomous%20Driving/Autonomous%20driving%20application%20-%20Car%20detection%20-%20v3.ipynb) （in Keras；框出图片中的车辆位置与大小，标注车辆类型；YOLO + Probability threshold filtering + Non-max suppression；未上传pre-trained YOLO）
4. [ResNet-50：数字手势识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/ResNets/Residual%20Networks%20-%20v2.ipynb) （in Keras；Identity / Convolutional block）
2. [CNN：笑脸识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Keras%20for%20Happy%20Face/Keras%20-%20Tutorial%20-%20Happy%20House%20v2.ipynb) （in Keras；ZeroPad + Conv2D + BatchNorm + ReLu + MaxPool）
3. [CNN：数字手势识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/CNN%20for%20Signs/Convolution%20model%20-%20Application%20-%20v1.ipynb) （in Tensorflow；[Conv2D + ReLu + MaxPool]*2）

### 深度神经网络 DNN

1. [三隐藏层：数字手势识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Tensorflow%20for%20Signs/Tensorflow%20Tutorial.ipynb) （in Tensorflow）
2. [多隐藏层：分辨图片是否为猫](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Deep%20Neural%20Network%20Application_%20Image%20Classification/Deep%20Neural%20Network%20-%20Application%20v3.ipynb) （in Numpy；研究隐藏层数对于结果的影响）
3. [单隐藏层：分辨二元图形](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Planar%20data%20classification%20with%20one%20hidden%20layer/Planar%20data%20classification%20with%20one%20hidden%20layer%20v4.ipynb) （in Numpy；研究隐藏单元数量对于结果的影响；[结果图像](https://raw.githubusercontent.com/YestinYang/Studying-Machine-Deep-Learning/master/img/single_layer_NN.png) ）
4. [以梯度下降训练逻辑回归模型：分辨图片是否为猫](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Logistic%20Regression%20as%20a%20Neural%20Network/Logistic%20Regression%20with%20a%20Neural%20Network%20mindset%20v4.ipynb) （预处理图片文件，用Numpy定义正反向传播算法、sigmoid激活函数和成本函数，训练模型并进行预测）


------

## 心得和总结

### 循环神经网络 RNN



### 卷积神经网络 CNN

1. [用Numpy构建卷积神经网络正向传播算法](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/CNN%20for%20Signs/Convolution%20model%20-%20Step%20by%20Step%20-%20v2.ipynb) （Zero-padding, Convolution, Max / Average pooling）

### 深度神经网络 DNN

1. [Keras使用指南及模型可视化](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Tensorflow_X_Keras_Building%20Blocks.ipynb) 
2. [Tensorflow使用指南及TensorBoard调用](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Tensorflow_Building_Blocks.ipynb) 
3. [不同梯度下降方法对深度神经网络的影响]() （Stochastic / Mini-batch gradient descent, Momentum, Adam）
4. [通过梯度检查确认算法正确](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Gradient%20Checking/Gradient%20Checking%20v1.ipynb) （Gradient directly estimated by cost function should be same as calculated by formulas）
5. [不同正则化方法对深度神经网络的影响](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Regularization/Regularization.ipynb) （None / L2 / Dropout）
6. [不同初始化方法对深度神经网络的影响](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Initialization/Initialization.ipynb) （Zero / Random / He initialization）
7. [用Numpy构建多隐藏层深度神经网络](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step%20v5.ipynb) 
8. [深度学习中的Numpy使用](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Python_Basics_with_Numpy/Python%20Basics%20With%20Numpy%20v3.ipynb) (Reshape, Normalization, Broadcasting, Vectorization)

### 机器学习

1. [决策树算法理论](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Machine%20Learning/Tree_Based_Algorithm_Related_Topics.ipynb)（随机森林原理，Bootstrap采样，OOB误差评估）

### Python基础

1. [Class面向对象编程](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/Class_OOP.ipynb) (Class / Instance Variable, Regular / Class / Static Method, Inheritance, Dunder Method, Decorators)
2. Python编程小实验
  - 创建网站介绍喜欢的电影 --> 进行中
  - [`turtle` 绘图](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/drawing_turtle.py) / [工作间隔休息提醒器](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/take_break.py) / [用词不当检测器](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/word_checker.py) / [批量文件自定义规则重命名](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/rename.py)


### 其他

1. [从Jupyter Notebook平台中打包并下载所有文件](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Download_Files_Jupyter_Hub.ipynb) 
2. [让Jupyter Notebook显示同个cell的多个outputs](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Anaconda%20Related/ipython_config.py) 
