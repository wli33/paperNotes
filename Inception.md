### v1
Motivation: improving the performance of deep neural networks by increasing their size: depth and width.

Problems:     
1. result a larger number of parameters: prone to overfitting     
2. increase use of computational resources
#### Architectural Details
![](https://hackathonprojects.files.wordpress.com/2016/09/naive.png?w=651&h=319)
![source:gitbub](https://user-images.githubusercontent.com/1249087/31683804-ea24827c-b34b-11e7-9934-eaf4fc80234a.png)  
Pool layer has no parameter.   
1×1 convolutions are used to compute reductions before the expensive 3×3 and 5×5 convolutions.    
Use average pooling instead of FC.   
To avoid gradient vanishing, add auxiliary classifiers connected to the intermediate layers during training and discard in test.   

Only 5 million params. Compared to AlexNet:   
- 12X less params   
- 2x more compute   
- 6.67% (vs. 16.4%)   

### v2-v3
1. use 2 3x3 to replace 5x5  
2. use 1xn and nx1 to nxn(n:12~20)   
(to be continued)
### Reference
[v1 Going Deeper with Convolutions, 6.67% test error](http://arxiv.org/abs/1409.4842)   
[v2 Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, 4.8% test error](http://arxiv.org/abs/1502.03167)    
[v3 Rethinking the Inception Architecture for Computer Vision, 3.5% test error](http://arxiv.org/abs/1512.00567)   
[v4 Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning, 3.08% test error](http://arxiv.org/abs/1602.07261)
