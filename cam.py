import torch
import torch.nn as nn
from torch.nn import functional as F

import cv2
import numpy as np



class CAM:
    '''
    Base Class
    '''
    def __init__(self, model, device, preprocess, layer_name=None):  
        if layer_name is None:
            self.layer_name = self.get_layer_name(model) 
        else:
            self.layer_name = layer_name
            
        self.model = model.to(device)
        self.device = device
        self.prep = preprocess
        self.feature = {}

        self.register_hook()
        
    def get_heatmap(self, img):
        pass
                                         
    def get_layer_name(self, model):
        layer_name = None

        for name, module in model.named_modules():
            if hasattr(module, 'inplace'):
                module.inplace = False

            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                layer_name = last_name
            last_name = name
        
        if layer_name is None:
            raise ValueError('Defaultly use the last layer before global average ' 
                             'pooling to plot heatmap. However, There is no such '
                             'layer in this model.\n'
                             'So you need to specify the layer to plot heatmap.\n'
                             'Arg "layer_name" is the layer you should specify.\n'
                             'Generally, the layer is deeper, the interpretaton ' 
                             'is better.')

        return layer_name

    def forward_hook(self, module, x, y):
        self.feature['output'] = y

    def register_hook(self):
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(self.forward_hook)
                break
        else:
            raise ValueError(f'There is no layer named "{self.layer_name}" in the model')

    def check(self, feature):
        if feature.ndim != 4 or feature.shape[2] * feature.shape[3] == 1:
            raise ValueError(f'Got invalid shape of feature map: {feature.shape}, '
                              'please specify another layer to plot heatmap.') 
                             
        

class GradCAM(CAM):
    def __init__(self, model, device, preprocess, layer_name=None):
        super().__init__(model, device, preprocess, layer_name)

    def get_heatmap(self, img):
        self.model.zero_grad()
        
        tensor = self.prep(img)[None, ...].to(self.device)
        output = self.model(tensor)
        feature = self.feature['output']
        self.check(feature)
        grad = torch.autograd.grad(output.max(1)[0], feature)[0]
        
        with torch.no_grad():        
            h, w = grad.size()[-2:]
            grad = grad.sum((2, 3), True) / (h * w)
            
            cam = (grad * feature).sum(1)
            F.relu(cam, True)
            cam = cam / cam.max() * 255
            cam = cam.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            cam = cv2.resize(cam, img.size[:2])
            cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)[..., ::-1]            
            
            if not isinstance(img, np.ndarray):
                img = np.asarray(img)

            overlay = np.uint8(0.6 * img + 0.4 * cam)
                    
        return output.detach(), overlay
                               
