# Learning Path of Machine Learning / Deep Learning / Data Analysis
内容以时间倒序（由近期到早期）排列。

## 学习清单

1. [Udacity Data Analyst (Advanced)](https://cn.udacity.com/course/data-analyst-nanodegree--nd002-cn-advanced) --> 进行中 (2018)
2. [Deep Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/deep-learning) --> [完成证书](https://www.coursera.org/account/accomplishments/specialization/certificate/MAJJ6QCYCYTX)  (2017-2018；笔记整理完毕)
   - 深度神经网络、卷积神经网络、递归神经网络、神经网络项目设计原则、数据集工程、模型误差分析
3. [Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning/) --> [完成证书](https://www.coursera.org/account/accomplishments/certificate/A4DF5DYNZENU) (2018；笔记整理完毕)
   - 逻辑回归，多元分类，神经网络，SVM，聚类与降维，异常检测，推荐算法，机器学习模型分析与评估
4. [Microsoft Professional Program in Data Science](https://www.edx.org/microsoft-professional-program-data-science) --> [完成证书](https://academy.microsoft.com/en-us/certificates/7539ddd1-5a3a-4bfe-9c0b-a2ed2bb42b8f/) (2016-2017；笔记整理中2/7)
   - 数据科学导论、Transact-SQL、Excel数据分析及可视化、数据科学的统计思维与分析、Python入门、数据科学基础、机器学习原理、Python数据科学编程、机器学习应用

> 上述课程的笔记与总结均可见于[技术博客](https://yestinyang.github.io/) 

------

## 项目列表

### 数据分析及数据挖掘

- `pandas` [采集评估清理分析WeRateDogs的Twitter数据](https://github.com/YestinYang/Learning-Path/blob/master/Projects/WeRateDogs/wrangle_act.ipynb) 
  - 基本思路（完成第二步）
    1. 数据采集：从本地、url、Twitter API三个数据源采集数据，并分别转化为`pandas.DataFrame` 
    2. 数据评估与清洗：满足质量与整洁度的要求
    3. 数据分析：对数据进行统计分析和可视化
    4. 完成分析报告
- `scikit-learn` `pandas` [预测美国大学毕业生收入水平](https://github.com/YestinYang/Learning-Path/blob/master/Projects/MS_Predict%20Student%20Earnings/Predict_Student_Earnings.ipynb) （[数据分析报告PDF](https://github.com/YestinYang/Learning-Path/blob/master/Projects/MS_Predict%20Student%20Earnings/Analysis%20of%20Student%20Earnings.pdf)）
  - 基本思路（已整理完成前四步）
    1. 必要的数据清洗，包括fill missing value, ordinal and one-hot encoding of categorical features
    2. 建立基准模型
    3. 根据基准模型的CV结果，迭代探索更多的数据清理和特征工程，包括remove high correlated features, interpolate with ExtraTreesRegressor for important feature
    4. 训练最优的单模型，评估模型包括`scikit-learn` 中的多个tree based models、XGBoost和LightGBM
    5. Stacking Models
  - [比赛项目介绍](https://datasciencecapstone.org/competitions/2/student-earnings/page/6/) 及[成绩排行榜](https://datasciencecapstone.org/competitions/2/student-earnings/leaderboard/) （ID为yestinyang88）

### 递归神经网络 RNN

- `Keras` [Many-to-many GRU：语音唤醒](https://github.com/YestinYang/Learning-Path/blob/master/Projects/Trigger%20word%20detection/Trigger%20word%20detection%20-%20v1.ipynb)  （CONV-1D + Batch Norm + ReLu + Dropout + [GRU + Dropout + Batch Norm]*2 + Dropout + Dense + Sigmoid；未上传train/validation dataset）
- `Keras` [Seq2Seq Attention Bi-LSTM for Translation：将多种格式的日期翻译为标准格式的日期](https://github.com/YestinYang/Learning-Path/blob/master/Projects/Machine%20Translation/Neural%20machine%20translation%20with%20attention%20-%20v3.ipynb) （Bi-LSTM as encoder + Attention + LSTM as decoder + softmax，其中Attention layer包括RepeatVector for last output of decoder + Concatenate to output of encoder + Dense + Softmax for weight + Dot for context of decoder；可视化Attention结果）
- `Keras` [Many-to-one LSTM：根据聊天文本自动选择合适的Emoji表情](https://github.com/YestinYang/Learning-Path/blob/master/Projects/Emojify/Emojify%20-%20v2.ipynb) （第一套方案为word vector均值 + Softmax；第二套方案为Embedding + [LSTM + Dropout]*2 + Softmax）
- `Keras` [One-to-many LSTM：爵士即兴作曲](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Jazz%20improvisation%20with%20LSTM/Improvise%20a%20Jazz%20Solo%20with%20an%20LSTM%20Network%20-%20v3.ipynb) （[结果声音文件](https://raw.githubusercontent.com/YestinYang/Studying-Machine-Deep-Learning/master/Projects/Jazz%20improvisation%20with%20LSTM/data/30s_trained_model.mp3)）
- `Numpy` [One-to-many RNN：生成风格一致的名字](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Dinosaur%20Island%20--%20Character-level%20language%20model/Dinosaurus%20Island%20--%20Character%20level%20language%20model%20final%20-%20v3.ipynb)（以恐龙的名字训练RNN，并逐字母生成新名字）

### 卷积神经网络 CNN

1. `Tensorflow` [VGG-19迁移学习：艺术风格化照片](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Neural%20Style%20Transfer/Art%20Generation%20with%20Neural%20Style%20Transfer%20-%20v2.ipynb) （将绘画作品的艺术风格迁移到日常照片中，合成新图片；成本函数 = 新旧照片的输出差异程度 + 新旧照片在数个选定卷积层的输出的格拉姆矩阵差异程度；[莫奈《亚嘉杜的罂粟花田》+ 法国罗浮宫训练过程图示](https://raw.githubusercontent.com/YestinYang/Studying-Machine-Deep-Learning/master/img/Art_Transfer_Procedure.png)；未上传pre-trained VGG-19）
2. `Keras` [FaceNet：人脸识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Face%20Recognition/Face%20Recognition%20for%20the%20Happy%20House%20-%20v3.ipynb) （1对N人脸匹配问题；Triplet loss function, L2 distance；未上传pre-trained FaceNet）
3. `Keras` [YOLO：自动驾驶中的车辆识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Car%20detection%20for%20Autonomous%20Driving/Autonomous%20driving%20application%20-%20Car%20detection%20-%20v3.ipynb) （框出图片中的车辆位置与大小，标注车辆类型；YOLO + Probability threshold filtering + Non-max suppression；未上传pre-trained YOLO）
4. `Keras` [ResNet-50：数字手势识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/ResNets/Residual%20Networks%20-%20v2.ipynb) （Identity / Convolutional block）
2. `Keras` [CNN：笑脸识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Keras%20for%20Happy%20Face/Keras%20-%20Tutorial%20-%20Happy%20House%20v2.ipynb) （ZeroPad + Conv2D + BatchNorm + ReLu + MaxPool）
3. `Tensorflow` [CNN：数字手势识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/CNN%20for%20Signs/Convolution%20model%20-%20Application%20-%20v1.ipynb) （[Conv2D + ReLu + MaxPool]*2）

### 深度神经网络 DNN

1. `Tensorflow` [三隐藏层：数字手势识别](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Tensorflow%20for%20Signs/Tensorflow%20Tutorial.ipynb) 
2. `Numpy` [多隐藏层：分辨图片是否为猫](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Deep%20Neural%20Network%20Application_%20Image%20Classification/Deep%20Neural%20Network%20-%20Application%20v3.ipynb) （研究隐藏层数对于结果的影响）
3. `Numpy` [单隐藏层：分辨二元图形](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Planar%20data%20classification%20with%20one%20hidden%20layer/Planar%20data%20classification%20with%20one%20hidden%20layer%20v4.ipynb) （研究隐藏单元数量对于结果的影响；[结果图像](https://raw.githubusercontent.com/YestinYang/Studying-Machine-Deep-Learning/master/img/single_layer_NN.png) ）
4. `Numpy` [以梯度下降训练逻辑回归模型：分辨图片是否为猫](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/Logistic%20Regression%20as%20a%20Neural%20Network/Logistic%20Regression%20with%20a%20Neural%20Network%20mindset%20v4.ipynb) （预处理图片文件，用Numpy定义正反向传播算法、sigmoid激活函数和成本函数，训练模型并进行预测）


------

## 心得和总结

### 递归神经网络 RNN

- `Numpy` [词语类比及去偏差](https://github.com/YestinYang/Learning-Path/blob/master/Deep%20Learning/Word%20Vector%20Representation/Operations%20on%20word%20vectors%20-%20v2.ipynb) （根据GloVe word embedding结果获取word vector, 计算Cosine similarity以完成 `a->b = c->?` 的类比；去除word vector的性别偏差）
- `Numpy` [用Numpy构建递归神经网络正向传播算法](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Building%20a%20Recurrent%20Neural%20Network%20-%20Step%20by%20Step/Building%20a%20Recurrent%20Neural%20Network%20-%20Step%20by%20Step%20-%20v2.ipynb) （RNN, LSTM cell and network）

### 卷积神经网络 CNN

1. `Numpy` [用Numpy构建卷积神经网络正向传播算法](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Projects/CNN%20for%20Signs/Convolution%20model%20-%20Step%20by%20Step%20-%20v2.ipynb) （Zero-padding, Convolution, Max / Average pooling）

### 深度神经网络 DNN

1. `Keras` [Keras使用指南及模型可视化](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Tensorflow_X_Keras_Building%20Blocks.ipynb) 
2. `Tensorflow` [Tensorflow使用指南及TensorBoard调用](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Tensorflow_Building_Blocks.ipynb) 
3. `Numpy` [不同梯度下降方法对深度神经网络的影响](https://github.com/YestinYang/Learning-Path/blob/master/Deep%20Learning/Optimization%20Methods/Optimization%20methods.ipynb) （Stochastic / Mini-batch gradient descent, Momentum, Adam）
4. `Numpy` [通过梯度检查确认算法正确](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Gradient%20Checking/Gradient%20Checking%20v1.ipynb) （Gradient directly estimated by cost function should be same as calculated by formulas）
5. `Numpy` [不同正则化方法对深度神经网络的影响](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Regularization/Regularization.ipynb) （None / L2 / Dropout）
6. `Numpy` [不同初始化方法对深度神经网络的影响](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Initialization/Initialization.ipynb) （Zero / Random / He initialization）
7. `Numpy` [用Numpy构建多隐藏层深度神经网络](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step%20v5.ipynb) 
8. `Numpy` [深度学习中的Numpy使用](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Deep%20Learning/Python_Basics_with_Numpy/Python%20Basics%20With%20Numpy%20v3.ipynb) （Reshape, Normalization, Broadcasting, Vectorization）

### 机器学习

1. [决策树算法理论](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Machine%20Learning/Tree_Based_Algorithm_Related_Topics.ipynb)（随机森林原理，Bootstrap采样，OOB误差评估，AdaBoost，GradientTreeBoosting）
2. `Matlab` 用Matlab实现机器学习算法 （[逻辑回归](https://github.com/YestinYang/Learning-Path/tree/master/Machine%20Learning/matlab1-Logistic%20Regression) ；[正则化](https://github.com/YestinYang/Learning-Path/tree/master/Machine%20Learning/matlab2-Regularization) ；[多元分类](https://github.com/YestinYang/Learning-Path/tree/master/Machine%20Learning/matlab3-Multi-Class%20Classification) ；[神经网络](https://github.com/YestinYang/Learning-Path/tree/master/Machine%20Learning/matlab4-Neural%20Network) ；[Bias vs Variance](https://github.com/YestinYang/Learning-Path/tree/master/Machine%20Learning/matlab5-Bias%20and%20Variance) ；[SVM支持向量机](https://github.com/YestinYang/Learning-Path/tree/master/Machine%20Learning/matlab6-SVM) ；[聚类与降维](https://github.com/YestinYang/Learning-Path/tree/master/Machine%20Learning/matlab7-Unsupervised) ；[异常检测与推荐](https://github.com/YestinYang/Learning-Path/tree/master/Machine%20Learning/matlab8-Anomaly%20and%20Recommendation) ）

### Python基础

1. [正则表达式与Python re模块](https://github.com/YestinYang/Learning-Path/blob/master/Basic%20Python/Regex_and_Python_re.ipynb) （regular expression / compile / finditer / findall / match / search）
2. [Python推导式](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/Python_Comprehensions.ipynb) （List / Dictionary / Set / Generator Comprehensions）
3. [Class面向对象编程](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/Class_OOP.ipynb) （Class / Instance Variable, Regular / Class / Static Method, Inheritance, Dunder Method, Decorators）
4. Python编程小实验
  - 创建网站介绍喜欢的电影 --> 进行中
  - [`turtle` 绘图](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/drawing_turtle.py) / [工作间隔休息提醒器](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/take_break.py) / [用词不当检测器](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/word_checker.py) / [批量文件自定义规则重命名](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Basic%20Python/rename.py)


### 其他

1. [从Jupyter Notebook平台中打包并下载所有文件](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Download_Files_Jupyter_Hub.ipynb) 
2. [让Jupyter Notebook显示同个cell的多个outputs](https://github.com/YestinYang/Studying-Machine-Deep-Learning/blob/master/Anaconda%20Related/ipython_config.py) 
