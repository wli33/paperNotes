From RCNN to fast-RCNN
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
2. Add SPP layer. The selected CNN features can send any size to SPP layer and it will produce fix size output. 
#### Fast RCNN
1. RoI pooling layer: a simplified one layer of SPP.
2. Multi-task model. loss has 2 parts: classification loss and regression loss for the Bbox. see [page 3 of the paper](https://arxiv.org/pdf/1504.08083.pdf)
![](http://img.blog.csdn.net/20160411154103099)

L1 loss is less sensitive to outliner than L2. grad(L1):1/-1; grad(L2): x to avoid grad exploding. smoothed L1: grad = x if |x|<i else 1/-1.

#### Faster RCNN
Add a Region Proposal Network (RPN) to produce region proposals directly; no need for external region proposals.
Use RoI Pooling to combine the proposal and feature map, send to upstream classifier and bbox regressor just like Fast R-CNN.

One network, four losses (Ross Girschick)
- RPN classification (anchor good / bad)
- RPN regression (anchor -> proposal)
- Fast R-CNN classification (over classes)
- Fast R-CNN regression (proposal -> box)
#### YOLO: You Only Look Once 
Divide image into S x S grid;      
Within each grid cell predict: B Boxes: 4 coordinates + confidence Class scores: C numbers;     
Direct prediction using a CNN;      
Faster than Faster R-CNN vgg16(FPS:45 vs 7), but not as good (mAP: 63.4 vs 73.2)

(To be continued)

Reference
----
Girschick et al, “Rich feature hierarchies for accurate object detection and semantic segmentation”, CVPR 2014         
Girschick, “Fast R-CNN”, ICCV 2015                 
Redmon et al, “You Only Look Once: Unified, Real-Time Object Detection”, arXiv 2015            
Ren et al, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks”, NIPS 2015      
Uijlings et al, “Selective Search for Object Recognition”, IJCV 2013
