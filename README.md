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
  * 用于loss曲线的绘制以及可视化生成图片
  * Used for drawing loss curve and visualization of generated pictures

## 使用指南(Guide)

### 训练(Training)

#### 1.准备数据集(Prepare dataset)

* dataset recommended：https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

* prepare your dataset：
  All the images should have 256\*256 px size with RGB channel
  The directory structure be like
  
```
    ├── datasets                   
    |   ├── <your_datasets_name>   # i.e. apple2orange         
    |   |   ├── trainA             # about 1000 images of apples
    |   |   ├── trainB             # about 1000 images of oranges
    |   |   ├── testA              # about 200 images of apples
    |   |   └── testB              # about 200 images of oranges
```

#### 2.开始训练(Start training)

* Do not use Colab：
    In your Python environment：(For example)
    ```
    python train.py --root datasets/<your_dataset_name>/ --cuda
    ```
    
    `--cuda` is optional, you can use GPU to accelerate your training speed.   
    
    You can also specify other arguments such as `--size 256` `--lr 0.0002`. Check out train.py for details.  
    
    If you want to view the loss function graph and the generated pictures, you can use **tensorboard**.
    ```
    tensorboard --logdir = logs
    ```
    Then you can click the URL http://localhost:6006/ and check the result.   
    
    During the training, checkpoints will be saved in folder `checkpoints`,

* Use Colab:
  []

### 测试(Testing)

In your Python environment：(For example)
```
python test.py --root datasets/<your_dataset_name>/ --model_root trained_model/<your_dataset_name>/model.pth --cuda
```
`--cuda` is optional. 

**model.pth** is the pretrained model which contains 2 generators' parameters.  

Test result will be saved in folders: **test_output/outputA** and **test_output/outputB**

### 应用(Apply)

Put the images you want to transfer into folder **apply/inA** or **apply/inB** (It depends on whether your images belongs to domain A or domain B.)  

In your Python enviroment: (For example)

```
python transfer.py --model_root trained_model/<your_dataset_name> --cuda
```

`--cuda` is optional.  

The result will be saved in folders: **apply/outputA** and **apply/outputB**  

**Notice:** All the results will be resized to 256\*256 px size. 





















