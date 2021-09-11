from PIL import Image
from os.path import join as pj
from pathlib import Path as p
import shutil, time
import numpy as np
from matplotlib import cm
import cv2 as cv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset import My_Dataset

# get the last layer before global average pooling layer
def get_layer_name(model):
    for n, m in model.named_children():
        # AdaptiveAvgPool2d or nn.AvgPool2d
        if isinstance(m, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
            name = tmp
        tmp = n
    return name

def f_hook(module, x, y):
    forward_hook["out"] = y

def b_hook(module, x, y):
    backward_hook["out"] = y

def main():
    data = 'beans'
    data = 'chijenxi_3c'
    data = 'rattle'
    md_name = "ExquisiteNetV2"
    #md_name = "mobilenetv3-large"
    #md_name = "ghostnet"
    heatmap_dir = "out"
    bs = 1 # batch size
    assert bs == 1
    core_num = 4
    ckp = "ckp" # weight_path
    pin_memory = True

    if p(heatmap_dir).exists():
        shutil.rmtree(heatmap_dir)
    """
    #let the batch size can be set to large
    #multiprocessing.set_sharing_strategy('file_system')
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    img_sets = dict((set_dir.name, My_Dataset(set_dir, T.ToTensor())) for set_dir in p(data).iterdir() if p(set_dir).is_dir())
    class_names = img_sets["train"].classes
    class_num = len(class_names)
    lb_dict = img_sets["train"].dict

    ds = dict((set_dir, DataLoader(img_sets[set_dir], batch_size=bs, shuffle=False, pin_memory=pin_memory, num_workers=core_num)) for set_dir in img_sets)
    ds_num = dict((set_dir, len(img_sets[set_dir])) for set_dir in img_sets)
    
    if md_name == "ExquisiteNetV2":
        model = ExquisiteNetV2(class_num)
        model.load_state_dict(torch.load(pj(ckp, data, "wt.pth"))['model'])
    model = model.to(device)
    model.eval()
    
    forward_hook = {"in":0, "out":0}
    backward_hook = {"in":0, "out":0}
    
    layer_name = get_layer_name(model)
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(f_hook)
            module.register_full_backward_hook(b_hook)

    for subset in ds:
        print("%s num:"%subset, ds_num[subset])
    
    print("you are using the {}!!!".format(device))
    
    time0 = time.time()
    correct = dict(zip(ds.keys(), np.zeros(len(ds))))
    for subset in ds:
        img_No = 1
        for inputs, labels, img_paths in ds[subset]:
            print(('%s_set: %d/%d'  %(subset, img_No, ds_num[subset])).ljust(25), end='\r')
            model.zero_grad()
            img_No += 1
            img_paths = img_paths[0]
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, pred_label = torch.max(outputs, 1)
            #outputs shape = 1x2
            outputs.squeeze_()[pred_label.item()].backward()
            with torch.set_grad_enabled(False):        
                feature_maps = forward_hook["output"]
                # backward hook is a tuple with one item
                grad_weights = backward_hook["output"][0]
                grad_weights = grad_weights.sum((2,3), True) / torch.prod(torch.as_tensor(grad_weights.shape[-2:]))
                heatmap = (grad_weights * feature_maps).sum(1)
                heatmap = nn.ReLU()(heatmap)
                heatmap /= torch.max(heatmap)
                heatmap = (heatmap * 255).to(dtype=torch.uint8, device="cpu")
            img = Image.open(img_paths)
            ori_img_shape = img.size
            if ori_img_shape != (224, 224):
                img.resize((224, 224), Image.LANCZOS)
            img = np.asarray(img)
            heatmap = heatmap.numpy().transpose(1,2,0)
            heatmap = cv.resize(heatmap, img.shape[:2], interpolation=4)
            heatmap = np.uint8(255 * cm.get_cmap("jet")(heatmap.squeeze()))
            img = np.uint8(0.5*img + 0.5*heatmap[:,:,:3])
            img = Image.fromarray(img)
            if ori_img_shape != (224, 224):
                img.resize(ori_img_shape, Image.LANCZOS)

            if pred_label == lb_dict[p(img_paths).parent.name]:
                dir_name = "bingo"
                correct[subset] += 1
            else:
                dir_name = "wrong"
            dst = list(p(img_paths).parts)
            dst.insert(1, dir_name)
            dst[0] = heatmap_dir
            dst = p("/".join(dst))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(img_paths, dst)
            img.save(dst.with_name(dst.stem+"_cam"+dst.suffix))
    acc = dict((subset, round(correct[subset]/ds_num[subset], 4))  for subset in ds)
    timez = time.time()
    print(" "*100) # overlap the last line which is printed on the screen
    for subset in ds:
        print("%s Acc:"%subset, acc[subset])
    print("Total spend:", timez-time0, "seconds")

if __name__ == '__main__':
    main()
