# MicroExpressionRecognition
使用casme2数据集训练的微表情识别MicroExpressionRecognition，支持摄像头、图片视频检测。
同时支持其他表情情绪识别数据集

![image](https://github.com/homxxx/MicroExpressionRecognition/blob/master/sdg2y-jxz4o.gif)

### 所需环境
tensorflow-gpu==1.13.1   
keras==2.1.5   

### 模型权重
casme2数据集训练的模型权重：

链接：https://pan.baidu.com/s/1nGRJRNc3EzSiVfrBnSPMkg 
提取码：klv1

也可自己申请数据集训练模型。

### 预测步骤

1. 按照训练步骤训练。  
2. 在classification.py文件里面，在如下部分修改model_path、classes_path、backbone和alpha使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类，backbone对应使用的主干特征提取网络，alpha是当使用mobilenet的alpha值**。  
3. recognition_camera.py 调用系统摄像头完成实时识别人脸微表情
4. recognition_video.py 视频检测
5. recognition_img.py 图片检测


### 训练步骤
1. datasets文件夹下存放的图片分为两部分，train里面是训练图片，test里面是测试图片。  
2. 在训练之前需要首先准备好数据集，在train或者test文件里里面创建不同的文件夹，每个文件夹的名称为对应的类别名称，文件夹下面的图片为这个类的图片。
3. 在准备好数据集后，需要在根目录运行txt_annotation.py生成训练所需的cls_train.txt，运行前需要修改其中的classes，将其修改成自己需要分的类。   
4. 之后修改model_data文件夹下的cls_classes.txt，使其也对应自己需要分的类。  
5. 在train.py里面调整自己要选择的网络和预权重后，就可以开始训练了！  
6. 预训练权重：
```
VGG-16：model_data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
mobilenet：model_data/mobilenet_2_5_224_tf_no_top.h5
```
8. 数据参考格式
```
|-datasets
    |-train
        |-disgust
            |-123.jpg
            |-234.jpg
        |-fear
            |-345.jpg
            |-456.jpg
        |-...
    |-test
        |-disgust
            |-567.jpg
            |-678.jpg
        |-fear
            |-789.jpg
            |-890.jpg
        |-...
```


### 评估步骤
1. datasets文件夹下存放的图片分为两部分，train里面是训练图片，test里面是测试图片，在评估的时候，我们使用的是test文件夹里面的图片。  
2. 在评估之前需要首先准备好数据集，在train或者test文件里里面创建不同的文件夹，每个文件夹的名称为对应的类别名称，文件夹下面的图片为这个类的图片。
3. 在准备好数据集后，需要在根目录运行txt_annotation.py生成评估所需的cls_test.txt，运行前需要修改其中的classes，将其修改成自己需要分的类。   
4. 之后在classification.py文件里面修改如下部分model_path、classes_path、backbone和alpha使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类，backbone对应使用的主干特征提取网络，alpha是当使用mobilenet的alpha值**。  
5. 运行eval_top1.py和eval_top5.py来进行模型准确率评估。

### Reference
https://github.com/keras-team/keras-applications   
https://github.com/bubbliiiing/classification-keras
