from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152


imagenet_model_dict = {
    "ResNet18": resnet18,
    "ResNet34": resnet34,
    "ResNet50": resnet50,
    "ResNet101": resnet101,
}
