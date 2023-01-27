# ArcFace with RepVGG backbone

ArcFace model with RepVGG as backbone.

Based on [Distributed Arcface Training in Pytorch](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) and [RepVGG](https://github.com/DingXiaoH/RepVGG).

## Verification Results

Taken from training log at final epoch.

```
[lfw][1618000]XNorm: 4.476631
[lfw][1618000]Accuracy-Flip: 0.99550+-0.00415
[lfw][1618000]Accuracy-Highest: 0.99583
[cfp_fp][1618000]XNorm: 4.102162
[cfp_fp][1618000]Accuracy-Flip: 0.93557+-0.01653
[cfp_fp][1618000]Accuracy-Highest: 0.93643
[agedb_30][1618000]XNorm: 4.434877
[agedb_30][1618000]Accuracy-Flip: 0.94733+-0.01373
[agedb_30][1618000]Accuracy-Highest: 0.95183
```

## Comparison with MobileFaceNet backbone

|      Backbone     | Forward/Backward pass size (MB) | Params size (MB) | Estimated Total Size (MB) | Total mult-adds (M) | Total params |
|:-----------------:|:-------------------------------:|:----------------:|:-------------------------:|:-------------------:|:------------:|
| MobileFaceNet     |              90.13              |       8.24       |           98.52           |        437.55       |   2,059,520  |
| RepVGG (training) |              16.24              |       33.94      |           50.33           |        387.72       |   8,484,352  |
| RepVGG (deploy)   |               3.63              |       30.74      |           34.52           |        349.45       |   7,684,768  |

## Citations
```
@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4690--4699},
  year={2019}
}
@inproceedings{ding2021repvgg,
  title={Repvgg: Making vgg-style convnets great again},
  author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han, Jungong and Ding, Guiguang and Sun, Jian},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={13733--13742},
  year={2021}
}
```
