# CloudTransfer-daytime2sunset-CycleGAN-Pytorch

本项目基于`Pytorch1.7.1`框架，实现了CycleGAN模型，与相应的训练、测试、应用部分功能.  

成功用于**白云<->晚霞**与**🍎苹果<->🍊橘子**图像风格转化。  
  
相较于原论文代码，本项目的代码更简洁、易读，个人认为比较适合入门者利用CycleGAN网络模型尝试图像风格迁移。  

**论文地址：** https://arxiv.org/abs/1703.10593  
  
**部分代码参考：** https://github.com/aitorzip/PyTorch-CycleGAN  (使用框架为`Pytorch 0.3`，部分代码在高版本Pytorch中已经失效)
<br/><br/><br/>
This project is based on `Pytorch1.7.1`, implements CycleGAN model, and the corresponding training, testing, and application functions.  
  
Successfully used in the image style transfer of **Cloud: daytime<->sunset** and **🍎apple<->🍊orange**.  

Compared with the original paper code, the code of this project is more easy to read. I personally think it is more suitable for beginners to use the CycleGAN network model to try image style transfer.

**Paper:** https://arxiv.org/abs/1703.10593  

**Part of the code reference:** https://github.com/aitorzip/PyTorch-CycleGAN    
(Based on `Pytorch 0.3`, part of the code is invalid in the higher version of Pytorch)  

## 结果展示(transfer result)

### 白云<->晚霞(cloud-daytime2sunset)

### 苹果<->橘子(apple2orange)

## 运行环境及部分安装包(Prerequisites)

* Python 3.8
* Pytorch(CUDA) 1.7.1
* torchvision 0.8.2
* numpy 1.20.2

* tensorboard 2.5.0 
  * 用于loss曲线的绘制以及生成图片的可视化
  * 

##


























