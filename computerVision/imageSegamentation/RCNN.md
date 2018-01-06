From RCNN to mask-RCNN
---------
#### RCNN
while reading paper [RCNN](https://arxiv.org/abs/1311.2524), I am not clear about how selection search works.
This is explained in [Selective Search for Object Recognition](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.414.1971&rep=rep1&type=pdf). 

[SelectiveSearch Algorithm](https://lilianweng.github.io/lil-log/assets/images/selective-search-algorithm.png)

The first step is based on [Efficient Graph-Based Image Segmentation](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf). The authors use graph theory to construct regional proposal.

[Segamentation Algorithm](http://img.blog.csdn.net/20140904111504850?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3VyZ2V3b25n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

The following are send to CNN. 
Note in cs231n, the detection part can also use "regression head" in the final layer. But it takes longer time for the loss to converge. Summary can be found [here](https://www.cnblogs.com/skyfsm/p/6806246.html).

#### SPP Net
Two improvements:    
1. Select Bbox on the CNN feature maps. speed 100x RCNN.  
b. Add SPP layer. The selected CNN features can send any size to SPP layer and it will produce fix size output. 
#### Fast RCNN
1. RoI pooling layer: a simplified one layer of SPP.
2. Multi-task model. loss has 2 parts: classification loss and regression loss for the Bbox.
![](http://img.blog.csdn.net/20160411154103099)

(To be continued)
