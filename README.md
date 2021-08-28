# Grad-CAM-pytorch
Original paper:  
[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://pytorch.org/)

# Important 1
Remember to specify the preprocess. (line 44 in `gradcam.py`)  
For example, if you nomalize every training data in the training phase, you should nomalize
the data before feeding them into the model to generate the heatmap.

# Important 2
In `gradcam.py`,  
variable `data` is the root directory of your dada.  
variable `layer_name` is the output of such layer which you want to use for generating heatmap. using the last layer before global averagepooling or last conv layer is in general.  
variable `heatmap_dir` is the directory in which the heatmap will save.

# Important 3
[ExquisiteNetV2](https://github.com/shyhyawJou/ExquisiteNetV2) is a model designed by me, you can find the code from the link.

if my code has fakes, don't hesitate to correct me. thx
