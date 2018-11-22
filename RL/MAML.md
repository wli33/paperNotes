Introduction
---------
An algorithm that is  general and model-agnostic.  
Train the model’s initial parameters such that the model has maximal performance on a new task after the parameters have been updated through one or more gradient steps computed with a small amount of data from that new task.  

For example, if we have two mutually exclusive tasks: go fwd and go bwd. You can't max reward for all the tasks but we can optimize rewards after a gradient step coz the gradient step gives your information about the task you suppose to perform.  

Advantages:  
does not expand the number of learned parameters nor place constraints on the model architecture.
broadly suitable for many tasks.
a small number of gradient updates will lead to fast learning on a new task.

Algorithms   
---------
Instead of supervised meta-learning(mapping (Dtrain,x)->y by rnn), use MAML mapping f(Dtrain,x)->y). MAML mapping f = (updated theta_i,x) for each task i.  

![](http://bair.berkeley.edu/blog/assets/maml/maml.png)  
![](https://cdn-images-1.medium.com/max/1600/1*_pgbRGIlmCRsYNBHl71mUA.png)  
Reference
---------
[Original paper](https://arxiv.org/pdf/1703.03400.pdf)  
[Github source code](https://github.com/cbfinn/maml)  
[Learning to Learn Berkeley AI Lab](https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/)  
[Paper repro: Deep Metalearning using “MAML” and “Reptile”](https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0)  
My thoughts:  
这篇paper看了一下午都还有不清楚的地方，加上之前面试面的不好，重新审视自己的能力是不是能达到这行的需要。
不过还是要继续下去，因为学习中的挫败是老朋友了。一方面要从基础开始循序渐进，不懂的一个个去查，是时间和耐心的活，另一方面就是坚持，好读书不求甚解。
现在找工作不太担心和迷茫未来，只在乎自己每天是不是在努力。
