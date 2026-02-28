import torch.nn as nn
import timm

def create_model(num_classes):
    model = timm.create_model("efficientnet_b0", pretrained=True)
    
    # Replace the final classification layer
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    
    return model