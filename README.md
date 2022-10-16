# Overview
The implementation of [Grad-CAM](https://arxiv.org/abs/1610.02391) for getting the attention map of CNN

# Update
> **[2021/12/03]**  
> optimize the speed of generating the heatmap

# Example
![](assets/n01669191_46.JPEG)
![](assets/heatmap.jpg)

# Usage
- The example image is generate from mobilenetv2:  
```
python show.py -d cpu -img n01669191_46.JPEG -layer features.18.0
```

- for custom model  
```
python show.py -d cpu -img n01669191_46.JPEG -layer {layer name} -m {your model path}
```

- If you have cuda, you can just replace the "cpu" to "cuda".
- If you don't specify any layer, my code will use the last layer before global average pooling  to plot heatmap.
- Get predict label  
  Very easy, you can refer to `show.py`.
