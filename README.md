# Update
> **[2021/12/03]**  
> optimize the speed of generating the heatmap

# Usage
My code is very easy to use

### step 1: create the GradCAM object
```
model = your_pytorch_model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
gradcam = GradCAM(model, device)
```
### step 2: get the heatmap
```
preprocess = your_preprocess
img = Image.open(img_path)  
img_tensor = preprocess(img).unsqueeze_(0).to(device)  
outputs, overlay = gradcam.get_heatmap(img, img_tensor)
overlay.show() # show the heatmap
```

# Complete Example
```
from PIL import Image

import toch
from torchvision import transfoms as T

from visualization import GradCAM


class_name = ['Class A', 'Class B']
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  
preprocess = T.Compose([
                        T.ToTensor()
                       ])  

gradcam = GradCAM(model, device) # create the GradCAM object  

img = Image.open(img_path)  
img_tensor = preprocess(img).unsqueeze_(0).to(device)  
outputs, overlay = gradcam.get_heatmap(img, img_tensor)
_, pred_label = outputs.max(1)
pred_class = class_name[pred_label.item()]
probability = F.softmax(outputs, 1).squeeze()[pred_label]

print("Result:", pred_class)
print("Probability:", probability)
overlay.show() # show the heatmap
```

# Reference
Original paper:  
[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)  

if you have any question, don't hesitate to contact me.  
