## 1. raw start
    时间关系训了三类0、1、2
    ep27左右，acc达到0.9左右，可视化散点图初步可分，但是分类边界不明显
    ep30左右，可以观察到散点在聚拢


## 2. train strategy
    后来发现应该先训no_center_loss的模型
    然后用center_loss模型fine-tune


## 3. cls center update
    1. using the GT label to embed a (1,2) vector standing for the center
        the center is updated along with the changement of the embedding layers' weights
        the entire layer is a black box to us

    2. using the batch center and a learning rate alpha to update the center
        the center is computed through the call function defined in the custom layer
        we know what actually happened through the pipeline