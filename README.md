# Learning Path of Machine Learning / Deep Learning / Data Analysis
内容以时间倒序（由近期到早期）排列。

## 学习清单

1. [Udacity Data Analyst (Advanced)](https://cn.udacity.com/course/data-analyst-nanodegree--nd002-cn-advanced) --> 进行中 (2018)
2. [Deep Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/deep-learning) --> [完成证书](https://www.coursera.org/account/accomplishments/specialization/certificate/MAJJ6QCYCYTX)  (2017-2018；笔记整理完毕)
3. [Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning/) --> [完成证书](https://www.coursera.org/account/accomplishments/certificate/A4DF5DYNZENU) (2018；笔记整理完毕)
4. [Microsoft Professional Program in Data Science](https://www.edx.org/microsoft-professional-program-data-science) --> [完成证书](https://academy.microsoft.com/en-us/certificates/7539ddd1-5a3a-4bfe-9c0b-a2ed2bb42b8f/) (2016-2017；笔记整理中2/7)

> 上述笔记均可见于[技术博客](https://yestinyang.github.io/) 

------

## 项目列表

### 递归神经网络 RNN

- [Many-to-many GRU：语音唤醒](https://github.com/YestinYang/Learning-Path/blob/master/Projects/Trigger%20word%20detection/Trigger%20word%20detection%20-%20v1.ipynb) `Keras` （CONV-1D + Batch Norm + ReLu + Dropout + [GRU + Dropout + Batch Norm]*2 + Dropout + Dense + Sigmoid；未上传train/validation dataset）
- [Seq2Seq Attention Bi-LSTM for Translation：将多种格式的日期翻译为标准格式的日期](https://github.com/YestinYang/Learning-Path/blob/master/Projects/Machine%20Translation/Neural%20machine%20translation%20with%20attention%20-%20v3.ipynb) `Keras` （Bi-LSTM as encoder + Attention + LSTM as decoder + softmax，其中Attention layer包括RepeatVector for last output of decoder + Concatenate to output of encoder + Dense + Softmax for weight + Dot for context of decoder；可视化Attention结果）
- [Many-to-one LSTM：根据聊天文本自动选择合适的Emoji表情](https://github.com/YestinYang/Learning-Path/blob/master/Projects/Emojify/Emojify%20-%20v2.ipynb) `Keras` （第一套方案为word vector均值 + Softmax；第二套方案为Embedding + [LSTM + Dropout]*2 + Softmax）
- [One-to-many LSTM：爵士即兴作曲](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Jazz%20improvisation%20with%20LSTM/Improvise%20a%20Jazz%20Solo%20with%20an%20LSTM%20Network%20-%20v3.ipynb) `Keras` （[结果声音文件](https://raw.githubusercontent.com/YestinYang/Studying-Machine-Deep-Learning/master/Projects/Jazz%20improvisation%20with%20LSTM/data/30s_trained_model.mp3)）
- [One-to-many RNN：生成风格一致的名字](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Dinosaur%20Island%20--%20Character-level%20language%20model/Dinosaurus%20Island%20--%20Character%20level%20language%20model%20final%20-%20v3.ipynb) `Numpy` （以恐龙的名字训练RNN，并逐字母生成新名字）

### 卷积神经网络 CNN

1. [VGG-19迁移学习：艺术风格化照片](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Neural%20Style%20Transfer/Art%20Generation%20with%20Neural%20Style%20Transfer%20-%20v2.ipynb) `Tensorflow` （将绘画作品的艺术风格迁移到日常照片中，合成新图片；成本函数 = 新旧照片的输出差异程度 + 新旧照片在数个选定卷积层的输出的格拉姆矩阵差异程度；[莫奈《亚嘉杜的罂粟花田》+ 法国罗浮宫训练过程图示](https://raw.githubusercontent.com/YestinYang/Studying-Machine-Deep-Learning/master/img/Art_Transfer_Procedure.png)；未上传pre-trained VGG-19）
2. [FaceNet：人脸识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Face%20Recognition/Face%20Recognition%20for%20the%20Happy%20House%20-%20v3.ipynb) `Keras` （1对N人脸匹配问题；Triplet loss function, L2 distance；未上传pre-trained FaceNet）
3. [YOLO：自动驾驶中的车辆识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Car%20detection%20for%20Autonomous%20Driving/Autonomous%20driving%20application%20-%20Car%20detection%20-%20v3.ipynb) `Keras` （框出图片中的车辆位置与大小，标注车辆类型；YOLO + Probability threshold filtering + Non-max suppression；未上传pre-trained YOLO）
4. [ResNet-50：数字手势识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/ResNets/Residual%20Networks%20-%20v2.ipynb) `Keras` （Identity / Convolutional block）
2. [CNN：笑脸识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Keras%20for%20Happy%20Face/Keras%20-%20Tutorial%20-%20Happy%20House%20v2.ipynb) `Keras` （ZeroPad + Conv2D + BatchNorm + ReLu + MaxPool）
3. [CNN：数字手势识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/CNN%20for%20Signs/Convolution%20model%20-%20Application%20-%20v1.ipynb) `Tensorflow` （[Conv2D + ReLu + MaxPool]*2）

### 深度神经网络 DNN

1. [三隐藏层：数字手势识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Tensorflow%20for%20Signs/Tensorflow%20Tutorial.ipynb) `Tensorflow` 
2. [多隐藏层：分辨图片是否为猫](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Deep%20Neural%20Network%20Application_%20Image%20Classification/Deep%20Neural%20Network%20-%20Application%20v3.ipynb) `Numpy` （研究隐藏层数对于结果的影响）
3. [单隐藏层：分辨二元图形](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Planar%20data%20classification%20with%20one%20hidden%20layer/Planar%20data%20classification%20with%20one%20hidden%20layer%20v4.ipynb) `Numpy` （研究隐藏单元数量对于结果的影响；[结果图像](https://raw.githubusercontent.com/YestinYang/Studying-Machine-Deep-Learning/master/img/single_layer_NN.png) ）
4. [以梯度下降训练逻辑回归模型：分辨图片是否为猫](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Logistic%20Regression%20as%20a%20Neural%20Network/Logistic%20Regression%20with%20a%20Neural%20Network%20mindset%20v4.ipynb) `Numpy` （预处理图片文件，用Numpy定义正反向传播算法、sigmoid激活函数和成本函数，训练模型并进行预测）


------

## 心得和总结

### 递归神经网络 RNN

- [词语类比及去偏差](https://github.com/YestinYang/Learning-Path/blob/master/Deep%20Learning/Word%20Vector%20Representation/Operations%20on%20word%20vectors%20-%20v2.ipynb) `Numpy` （根据GloVe word embedding结果获取word vector, 计算Cosine similarity以完成 `a->b = c->?` 的类比；去除word vector的性别偏差）
- [用Numpy构建递归神经网络正向传播算法](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Building%20a%20Recurrent%20Neural%20Network%20-%20Step%20by%20Step/Building%20a%20Recurrent%20Neural%20Network%20-%20Step%20by%20Step%20-%20v2.ipynb) `Numpy` （RNN, LSTM cell and network）

### 卷积神经网络 CNN

1. [用Numpy构建卷积神经网络正向传播算法](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/CNN%20for%20Signs/Convolution%20model%20-%20Step%20by%20Step%20-%20v2.ipynb) `Numpy` （Zero-padding, Convolution, Max / Average pooling）

### 深度神经网络 DNN

1. [Keras使用指南及模型可视化](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Tensorflow_X_Keras_Building%20Blocks.ipynb) `Keras` 
2. [Tensorflow使用指南及TensorBoard调用](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Tensorflow_Building_Blocks.ipynb) `Tensorflow` 
3. [不同梯度下降方法对深度神经网络的影响](https://github.com/YestinYang/Learning-Path/blob/master/Deep%20Learning/Optimization%20Methods/Optimization%20methods.ipynb) `Numpy` （Stochastic / Mini-batch gradient descent, Momentum, Adam）
4. [通过梯度检查确认算法正确](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Gradient%20Checking/Gradient%20Checking%20v1.ipynb) `Numpy` （Gradient directly estimated by cost function should be same as calculated by formulas）
5. [不同正则化方法对深度神经网络的影响](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Regularization/Regularization.ipynb) `Numpy` （None / L2 / Dropout）
6. [不同初始化方法对深度神经网络的影响](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Initialization/Initialization.ipynb) `Numpy` （Zero / Random / He initialization）
7. [用Numpy构建多隐藏层深度神经网络](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step%20v5.ipynb) `Numpy` 
8. [深度学习中的Numpy使用](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Python_Basics_with_Numpy/Python%20Basics%20With%20Numpy%20v3.ipynb) `Numpy` （Reshape, Normalization, Broadcasting, Vectorization）

### 机器学习

1. [决策树算法理论](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Machine%20Learning/Tree_Based_Algorithm_Related_Topics.ipynb)（随机森林原理，Bootstrap采样，OOB误差评估）
2. 用Matlab实现机器学习算法 `Matlab` （[逻辑回归](https://github.com/YestinYang/Learning-Path/tree/master/Machine%20Learning/matlab1-Logistic%20Regression) ；[正则化](https://github.com/YestinYang/Learning-Path/tree/master/Machine%20Learning/matlab2-Regularization) ；[多元分类](https://github.com/YestinYang/Learning-Path/tree/master/Machine%20Learning/matlab3-Multi-Class%20Classification) ；[神经网络](https://github.com/YestinYang/Learning-Path/tree/master/Machine%20Learning/matlab4-Neural%20Network) ；[Bias vs Variance](https://github.com/YestinYang/Learning-Path/tree/master/Machine%20Learning/matlab5-Bias%20and%20Variance) ；[SVM支持向量机](https://github.com/YestinYang/Learning-Path/tree/master/Machine%20Learning/matlab6-SVM) ；[聚类与降维](https://github.com/YestinYang/Learning-Path/tree/master/Machine%20Learning/matlab7-Unsupervised) ；[异常检测与推荐](https://github.com/YestinYang/Learning-Path/tree/master/Machine%20Learning/matlab8-Anomaly%20and%20Recommendation) ）

### Python基础

1. [Python推导式](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/Python_Comprehensions.ipynb) （List / Dictionary / Set / Generator Comprehensions）
2. [Class面向对象编程](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/Class_OOP.ipynb) （Class / Instance Variable, Regular / Class / Static Method, Inheritance, Dunder Method, Decorators）
2. Python编程小实验
  - 创建网站介绍喜欢的电影 --> 进行中
  - [`turtle` 绘图](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/drawing_turtle.py) / [工作间隔休息提醒器](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/take_break.py) / [用词不当检测器](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/word_checker.py) / [批量文件自定义规则重命名](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/rename.py)


### 其他

1. [从Jupyter Notebook平台中打包并下载所有文件](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Download_Files_Jupyter_Hub.ipynb) 
2. [让Jupyter Notebook显示同个cell的多个outputs](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Anaconda%20Related/ipython_config.py) 
