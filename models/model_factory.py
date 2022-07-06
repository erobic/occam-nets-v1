import logging
from models.variable_width_resnet import *
from models.occam_resnet import *
# from models.occam_densenet import *
# from torchvision.models.efficientnet import *
# from models.occam_efficient_net import *
# from torchvision.models.mobilenet import *
# from models.occam_mobile_net import *


def build_model(model_config):
    if 'occam' in model_config.name:  # or 'resnet' in model_config.name:
        m = eval(model_config.name)(model_config.num_classes)
    elif 'efficientnet_b' in model_config.name:
        m = eval(model_config.name)(pretrained=False)
        m.classifier = nn.Sequential(
            nn.Dropout(p=m.classifier[0].p, inplace=True),
            nn.Linear(m.classifier[1].in_features, model_config.num_classes),
        )
    elif 'mobilenet' in model_config.name:
        m = eval(model_config.name)(pretrained=False)
        m.classifier[-1] = nn.Linear(m.classifier[0].out_features, model_config.num_classes)
        if 'img64' in model_config.name:
            conv = m.features[0][0]
            m.features[0][0] = nn.Conv2d(conv.in_channels, conv.out_channels,
                                         kernel_size=(conv.kernel_size, conv.kernel_size),
                                         stride=(1, 1),
                                         padding=(1, 1), bias=False)
    else:
        m = eval(model_config.name)(num_classes=model_config.num_classes)
        if model_config.num_classes != m.fc.out_features:
            m.fc = nn.Linear(m.fc.in_features, model_config.num_classes)

    logging.getLogger().debug("Model: ")
    logging.getLogger().debug(m)
    return m
