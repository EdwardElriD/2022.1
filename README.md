# 2022.1
TinySSD

一、测试代码

1.文件下带有该文件的main函数，直接运行可以可视化模块如何改变数据的形状。

2.直接运行train.py重新训练模型，模型的训练结果将会以每10轮的形式保存。

3.test.py能够对给出的test数据进行测试，可以通过调用训练好的模型权重来观察测试效果（net_30.pkl）。运行时需要注意更改测试图片的路径，根据测试环境实际路径进行更改。

二、模块功能

1.data_load.py：从指定的文件夹中读取所需训练数据，制作成训练所需的数据类型。

2.gtboxes_and_bbx.py: 得到边界框与锚框。

3.anchors.py： 生成以每个像素为中心具有不同形状的锚框。

4.iou.py：计算锚框与锚框或边界框列表中成对的交并比。

5.loss.py：计算标签损失(交叉熵损失函数)和类别损失(L1损失函数)。

6.nms.py：找出含有物体的锚框，采用置信度和非极大值抑制置信度筛选出最终需要的锚框(预测阶段使用)

7.prediction.py：定义类别和标签的预测函数，由于存在多个预测输出，需要先将预测结果压平，再连接起来

8.tinySSD.py：TinySSD网络搭建

9.blocks.py：定义下采样块，基础网络块，以及后续块的整合方法

三、数据增强

1.在data_load中进行了数据增强，对原有的数据使用了ColorJitter调整了亮度、对比度、饱和度和色调。同时扩充数据集大小至2000.
数据增强后的测试结果如augmentation.jpg所示，将置信度提高至0.97，较数据增强前的置信度（见original.jpg）提高了0.14,效果显著.


四、注意事项

1.根据实际采用CPU/GPU训练模型时需要注意调整train.py与test.py中的device参数，以匹配实际训练环境。

2.当训练轮数大于十的倍数时才会保存训练权重，如epoch为20时仅保存net_10.pkl，当epoch=21时才会保存至net_20.pkl。
