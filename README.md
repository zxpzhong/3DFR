# Differential Render based 3D Finger Reconstruction
基于可微渲染器的三维手指重建

# Requirements

- Python \>= 3.5 (3.6 recommended)
- Training : [pytorch](https://github.com/pytorch/pytorch)>=1.0
- Deploy：[ncnn](https://github.com/Tencent/ncnn)
- [Protobuf](https://gist.github.com/diegopacheco/cd795d36e6ebcd2537cd18174865887b)
- torchvision>=0.4.0
- tqdm

# 使用方法

训练：`python train.py -c config.json`

# 文件夹目录

.  
├── base ：储存data_loader，网络，训练器的基类  
├── **config.json**：配置文件  
├── data：数据csv文件夹  
├── data_loader：data_loader具体实现  
├── logger：日志器代码实现  
├── loss：损失  
├── **model**：网络模型  
├── parse_config.py：参数压缩加载  
├── requirements.txt：依赖  
├── saved：保存的模型、代码、日志都会在这  
├── test.py：测试代码接口  
├── **trainer**：训练器代码实现  
├── **train.py**：训练代码接口  
└── utils：小组件代码  

# 如何更改训练网络

1. 在model文件夹中添加网络代码
2. 网络类继承自`nn.Model`类
   在forward方法中实现输入图片x，输出logit的功能
   在extract_feature方法中实现输入图片x，输出特征f的功能
3. 在`model/model.py`中添加`from model.文件名 import 网络类名`
4. 在config.json中修改`arch`字段的`type`属性为网络类名

# 如何更改训练数据集

1. 训练文件和测试csv文件都在`data/csv`下
2. 训练csv文件格式为: 表头:`number,flag,img_path,label`,分别表示序号,是否为训练,图片路径和subject标签
3. 测试csv文件格式为: 表头:`number,flag,img1_path,img2_path`,分别表示序号,是否为类内样本,图片1路径,图片2路径
4. 制作好对应的csv后,在`config.json`中修改`data_loader/args`字段的`data_dir`和`test_dir`属性
5. 修改`arch/args`下的`num_classes`属性为训练集中的类别数

# 如何更改train/test

修改`trainer/trainer.py`中`_train_epoch`和`_valid_epoch`

