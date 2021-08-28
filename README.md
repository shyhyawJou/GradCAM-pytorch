# Grad-CAM-pytorch
Original paper:  
[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://pytorch.org/)

# Important 1
Remember to specify the preprocess. (line 49 in `gradcam.py`)  
For example, if you nomalize every training data in the training phase, you should nomalize
the data before feeding them into the model to generate the heatmap.

# Important 2
[ExquisiteNetV2](https://github.com/shyhyawJou/ExquisiteNetV2) is a model designed by me, you can find the code from the link.
