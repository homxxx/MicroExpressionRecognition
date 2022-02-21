from .mobilenet import MobileNet
from .resnet50 import ResNet50
from .vgg16 import VGG16

get_model_from_name = {
    "mobilenet"     : MobileNet,
    "resnet50"      : ResNet50,
    "vgg16"         : VGG16,
}

freeze_layers = {
    "mobilenet"     : 81,
    "resnet50"      : 173,
    "vgg16"         : 19,
    "cspdarknet53"  : 60,
}
