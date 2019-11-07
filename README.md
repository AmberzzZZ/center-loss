## 1. raw start
    时间关系训了三类0、1、2
    ep27左右，acc达到0.9左右，可视化散点图初步可分，但是分类边界不明显
    ep30左右，可以观察到散点在聚拢

## 2. train strategy
    后来发现应该先训no_center_loss的模型
    然后用center_loss模型fine-tune


