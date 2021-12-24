import torch
import torch.nn as nn
from torch.nn import functional as F

from PIL import Image
import cv2 as cv
from matplotlib import cm
import numpy as np


class GradCAM:
    def __init__(self, model, device):
        layerName_for_visualization = self.get_layer_name(model)
        for name, layer in model.named_children():
            layer.requires_grad_(False)
            if name == layerName_for_visualization:
                layer.requires_grad_(True)
                layer.register_forward_hook(self.forward_hook)
                layer.register_full_backward_hook(self.backward_hook)
                break
            
        self.model = model
        self.device = device
        self.featuremaps = {}
        self.gradients = {}

    def get_heatmap(self, img, img_tensor):
        self.model.zero_grad()
        img_tensor = img_tensor.to(self.device)
        outputs = self.model(img_tensor)
        _, pred_label = outputs.max(1)
        #outputs shape = 1x2
        outputs[0][pred_label].backward()
        with torch.no_grad():        
            feature_maps = self.featuremaps["output"]
            # "gradients" is a tuple with one item
            grad_weights = self.gradients["output"][0]
            h, w = grad_weights.size()[-2:]
            grad_weights = grad_weights.sum((2,3), True) / (h * w)
            heatmap = (grad_weights * feature_maps).sum(1)
            F.relu(heatmap, True)
            heatmap /= heatmap.max()
            heatmap = (heatmap * 255).to(dtype=torch.uint8, device="cpu")
            heatmap = heatmap.numpy().transpose(1,2,0)
            heatmap = cv.resize(heatmap, img.size[:2], interpolation=4)
            heatmap = np.uint8(255 * cm.get_cmap("jet")(heatmap.squeeze()))

            if not isinstance(img, np.ndarray):
                img = np.asarray(img)
            img_size = img.shape[:2][::-1] # w, h

            overlay = np.uint8(0.6*img + 0.4*heatmap[:,:,:3])
            overlay = Image.fromarray(overlay)
            if overlay.size != img_size:
                overlay = overlay.resize(img_size, Image.BILINEAR)

        return outputs.detach(), overlay

    def get_layer_name(self, model):
        for n, m in model.named_children():
            # AdaptiveAvgPool2d or nn.AvgPool2d
            if isinstance(m, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                name = tmp
            tmp = n
        return name

    def forward_hook(self, module, x, y):
        self.featuremaps["input"] = x
        self.featuremaps["output"] = y

    def backward_hook(self, module, x, y):
        self.gradients["input"] = x
        self.gradients["output"] = y
        
        
