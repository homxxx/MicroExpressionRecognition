#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from nets.mobilenet import MobileNet
from nets.resnet50 import ResNet50
from nets.vgg16 import VGG16

if __name__ == "__main__":
    model = MobileNet([224, 224, 3], classes=1000)
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i,layer.name)
