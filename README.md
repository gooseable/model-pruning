# model-pruning
对Yolov3进行模型剪枝达到加速目地.

参考资料：
https://github.com/Lam1360/YOLOv3-model-pruning

https://github.com/talebolano/yolov3-network-slimming

https://github.com/tanluren/yolov3-channel-and-layer-pruning


基于论文 Learning Efficient Convolutional Networks Through Network Slimming (ICCV 2017)的 channel pruning算法

环境：
python36  torch1.0.0



正常训练（Baseline）

python train.py --model_def config/yolov3-hand.cfg


剪枝算法的步骤

    进行稀疏化训练

    python train.py --model_def config/yolov3-hand.cfg -sr --s 0.01


    基于 test_prune.py 文件进行剪枝（通过设定合理的剪枝规则），得到剪枝后的模型

    对剪枝后的模型进行微调（本项目对原算法进行了改进，实验过程中发现即使不经过微调也能达到较高的 mAP）

    python train.py --model_def config/prune_yolov3-hand.cfg -pre checkpoints/prune_yolov3_ckpt.pth


中间遇到的问题：

1.当版本为3.5时，train.py报错，metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

2.改为3.6版本发现报错：
  File "train.py", line 78, in <module>
    model.load_darknet_weights(opt.pretrained_weights)
  File "/home/cct/yolo_optimizer/YOLOv3-model-pruning/models.py", line 318, in load_darknet_weights
    conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
RuntimeError: shape '[256, 128, 3, 3]' is invalid for input of size 160590

原因是使用的权重文件不完整，换完整版的yolov3.weights,这个错误消失。

3.还遇到了一些缺文件、函数的错误，重新在更新了函数错误消失。


