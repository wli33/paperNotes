From RCNN to Mask-RCNN
---------
#### RCNN
while reading paper [RCNN](https://arxiv.org/abs/1311.2524), I am not clear about how selection search works.
This is explained in [Selective Search for Object Recognition](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.414.1971&rep=rep1&type=pdf). 

[SelectiveSearch Algorithm](https://lilianweng.github.io/lil-log/assets/images/selective-search-algorithm.png)

The first step is based on [Efficient Graph-Based Image Segmentation](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf). The authors use graph theory to construct regional proposal.

[Segamentation Algorithm](http://img.blog.csdn.net/20140904111504850?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3VyZ2V3b25n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

First get ~2k proposed images and labels in training set. If IoU<threshold, reset Y class = bg instead of given label.
In AlexNet threshold = 0.5 for fine-tune, 0.3 for svm.
```
index = int(tmp[1]) # tmp1 = label in txt file
if svm == False:
    label = np.zeros(num_clss+1)
    if iou_val < threshold: label[0] = 1
    else: label[index] = 1
    labels.append(label)
else:
    if iou_val < threshold: labels.append(0)
    else: labels.append(index)
```
The following are sent to CNN (use SVM as final layer rather than softmax). e.g. Fc7:2k * 4096 features, Weights_svm: 4096* C, final result: 2k * C.
```
    img_path = 'testimg7.jpg'
    #get proposals
    imgs, verts = image_proposal(img_path) 
    net = create_alexnet(3) # num_classes:3 including bg class 0
    model = tflearn.DNN(net)
    model.load('fine_tune_model_save.model')
    svms = train_svms(train_file_folder, model)
    features = model.predict(imgs)
    
    for f in features:
        for i in svms:
	    pred = i.predict(f)
```


#### SPP Net
crops and wrap cause loss of info or distortion. Fixed-size only needs for fc layer.
Two improvements:    
1. Select Bbox on the CNN feature maps. speed 100x RCNN.        
2. Add SPP layer. The selected CNN features can send any size to SPP layer and it will produce fix size output.  
From conv5->fc6: size:axa(13x13) n=pool(3x3) win = ceil(a/n) and stride str = floor(a/n). total:15xnum of filer(256)    

pool3x3      | pool2x2     | pool1x1     |    fc6    
-------------|-------------|-------------|--------------
type=pool    | type=pool   | type=pool   |   type=fc    |
pool=max     | pool=max    | pool=max    |   outputs=4096 |
inputs=conv5 | inputs=conv5| inputs=conv5|   inputs=pool3x3,pool2x2,pool1x1/filter|
sizeX=5      | sizeX=7     | sizeX=13    |
stride=4     | stride=6    | stride=6    |

input->conv5->get ~2k proposal feature maps->SPP layer->fc->svm

Mapping a Window to Feature Maps:  
(x,y)=(S*x’,S*y’)  S = products of all strides in conv and pool layer, pad floor(p/2) pixels for a layer with a filter size of p  
so left,top: x' = floor(x/S)+1   right, bottom：x' = ceil(x/S)-1 
If the padding is not floor(p/2), need to add a proper offset to x.

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

#### Mask-RCNN
![](http://img.blog.csdn.net/20170614225558493)      
1.Roi Pooling->RoiAlign     
2. Add Loss for mask      
(To be continued)

Reference
----
Girschick et al, “Rich feature hierarchies for accurate object detection and semantic segmentation”, CVPR 2014         
Girschick, “Fast R-CNN”, ICCV 2015                 
Redmon et al, “You Only Look Once: Unified, Real-Time Object Detection”, arXiv 2015            
Ren et al, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks”, NIPS 2015      
Uijlings et al, “Selective Search for Object Recognition”, IJCV 2013
