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
1. add BN layer
2. use 2 3x3 to replace 5x5; use 1xn and nx1 to replace nxn(n:12~20)  

BN: Accelerate Training by Reducing Internal Covariate Shift, 14 times fewer training steps. It also acts as a regularizer, in some cases eliminating the need for Dropout. 

v3: 3,5,2 incetion layers

#### Model Regularization via Label Smoothing  
Motivation:1. prevent overfitting   2. increase the ability of the model to adapt(prevent model from becoming to confident)   
How: 
1. set it to the groundtruth label k = y; 
2. with probability e, replace k with a sample drawn from the distribution u(k). they used the uniform distribution u(k) = 1/K  

### v4
Reduction achieved by valid padding.  
V4: a pure Inception variant without residual connections with roughly the same recognition performance as Inception-ResNet-v2. 4,7,3 inception layers. more stack layers in the inception block.  
"the step time of Inception-v4 proved to be significantly slower in practice, probably due to the larger number of layers."

Inception-ResNet-v1: a hybrid Inception version that has a similar computational cost to Inception-v3. 5,10,5 inception layers.

Inception-ResNet-v2: a costlier hybrid Inception version with significantly improved recognition performance. 5,10,5 inception layers. 
#### Scaling of the Residuals
if the number of filters > 1000, the residual variants started to exhibit instabilities and the network has just “died” early in the training. the last layer before the average pooling started to produce only 0 after a few tens of thousands of iterations.
This could not be prevented by lowering the learning rate nor by adding an extra BN to this layer.
![](https://qph.fs.quoracdn.net/main-qimg-c4940ebeff4ccc7704e2596b435b2f25)   
scaling down the residuals before adding them to the previous layer activation seemed to stabilize the training. scaling factors: 0.1- 0.3 
(to be continued)
### Reference
[v1 Going Deeper with Convolutions, 6.67% test error](http://arxiv.org/abs/1409.4842)   
[v2 Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, 4.8% test error](http://arxiv.org/abs/1502.03167)    
[v3 Rethinking the Inception Architecture for Computer Vision, 3.5% test error](http://arxiv.org/abs/1512.00567)   
[v4 Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning, 3.08% test error](http://arxiv.org/abs/1602.07261)
