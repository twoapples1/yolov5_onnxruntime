# YoloV5

## 论文

无

## 模型结构

YoloV5是一种单阶段目标检测算法，该算法在YOLOV4的基础上添加了一些新的改进思路，使其速度与精度都得到了极大的性能提升。

<img src=./doc/YOLOV5_01.jpg style="zoom:100%;" align=middle>

## 算法原理

YOLOv5算法通过将图像划分为不同大小的网格，预测每个网格中的目标类别和边界框，利用特征金字塔结构和自适应的模型缩放来实现高效准确的实时目标检测。

## 环境安装

step1:安装opencv

```bash
#安装opencv4.6
wget https://github.com/opencv/opencv/archive/refs/tags/4.6.0.tar.gz
cd opencv
mkdir build
cd build
sudo cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make
make install
```

step2:克隆项目并修改本地ORT所在目录并编译

```bash
git clone xxx.git
mkdir build
cmake ..
make
./YOLOV5
```

## 效果图展示

### FP32

<div style="text-align:left;">
  <img src="./resource/images/result.jpg" alt="Image" style="width:500px;">
</div>

## 应用场景

### 算法类别

`目标检测`

### 热点应用行业

`交通`,`教育`,`化工`

## 参考资料

https://github.com/ultralytics/yolov5
https://github.com/itsnine/yolov5-onnxruntime
