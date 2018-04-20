#### v1
Motivation: improving the performance of deep neural networks by increasing their size: depth and width.

Problems:     
1. result a larger number of parameters: prone to overfitting     
2. increase use of computational resources
#### Architectural Details
![](https://hackathonprojects.files.wordpress.com/2016/09/naive.png?w=651&h=319)
![source:gitbub](https://user-images.githubusercontent.com/1249087/31683804-ea24827c-b34b-11e7-9934-eaf4fc80234a.png)  
1×1 convolutions are used to compute reductions before the expensive 3×3 and 5×5 convolutions.

(to be continued)
### Reference
[v1 Going Deeper with Convolutions, 6.67% test error](http://arxiv.org/abs/1409.4842)
