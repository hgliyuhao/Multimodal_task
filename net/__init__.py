from .mobilenet import MobileNet
from .resnet50 import ResNet50
from .vgg16 import VGG16
from .vit import VisionTransformer



def build_cv_model(model_name,hidden_size = 64):
    
    input_shape = [224,224]
    
    get_model_from_name = {
        "mobilenet"     : MobileNet,
        "resnet50"      : ResNet50,
        "vgg16"         : VGG16,
        "vit"           : VisionTransformer
    }
     
    model = get_model_from_name[model_name](input_shape=[input_shape[0], input_shape[1], 3], classes = hidden_size)
    
    return model
    