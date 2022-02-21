from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.models import Model        #导入包Conv2D是卷积核 Flatten是展开 Input输入  MaxPooling2D最大卷积核


def VGG16(input_shape=None, classes=1000): #def 就是开始定义VGG16的网络
    img_input = Input(shape=input_shape)  # 224, 224, 3

    # Block 1
    # 224, 224, 3 -> 224, 224, 64
    x = Conv2D(64, (3, 3),  #开始第一个卷积核进行特征提取，64为卷积核的个数；(3, 3)是卷积核的大小
                      activation='relu', #relu激活函数
                      padding='same',  #padding='same' 尺寸不变
                      name='block1_conv1')(img_input)  #dui juanjihe jinxing mingming  # x= 224*224*64
    x = Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)    #224*224*64

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)  #112*122*64

    # Block 2

    x = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x) #112*112*128
    x = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3

    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4

    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
                      

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5

    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    # 14, 14, 512 -> 7, 7, 512
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x) #激活函数

    inputs = img_input

    model = Model(inputs, x, name='vgg16')
    return model

if __name__ == '__main__':
    model = VGG16(input_shape=(224, 224, 3))
    model.summary()
