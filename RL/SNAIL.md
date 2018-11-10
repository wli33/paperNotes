A SIMPLE NEURAL ATTENTIVE META-LEARNER
---------
Deep neural networks tend to struggle:   
1. when data is scarce;  
2. when need to adapt quickly to changes in the task.  
Solution: training a meta-learner on a distribution of similar tasks, hope to generalize to novel but related
tasks by learning a high-level strategy that captures the essence of the problem it is asked to solve.

Many recent meta-learning approaches are extensively hand-designed:   
architectures specialized to a particular application, or hard-coding algorithmic components that constrain how the meta-learner solves the task. 

#### Architecture
The author propose Simple Neural AttentIve Learner (or SNAIL) that use a combination of convolutions and soft attention layers. 

![](https://lilianweng.github.io/lil-log/assets/images/snail.png)
In reinforcement-learning settings, it receives a sequence of observation-action-reward tuples.  
consist of dense block, TC block and attention block:  
![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/7e9c1e0d247b20a0683f4797d9ea248c3b53d424/7-Figure2-1.png)

#### Experiments
1. Image classification.  
Dataset: 5-way Omniglot, 20-way Omniglot, and 5-way mini-ImageNet. 
sample N classes from the dataset and K examples of each class, followed by a new, unlabeled example from one of the N classes.  
report the average accuracy on this last, (NK + 1)-th timestep. 

2. Reinforcement learning(MULTI-ARMED BANDITS,TABULAR MDPS,CONTINUOUS CONTROL, visual navigation).
Learning to visually navigate a maze  
- train on 1000 small mazes  
- test on held-out small mazes and large mazes   
![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTh57tTx4aXbu1rY8lPi5QzHjbLoqNqYS1nkoY_2QDcyqWk-QSiWw)  
Results: fast time to find goal.

A few Notes:  
Dilated convolutions([paper by Fisher Yu and Vladlen Koltun](https://arxiv.org/abs/1511.07122)): dilation of 0: w[0]*x[0] + w[1]*x[1] + w[2]*x[2]. dilation of 1:w[0]*x[0] + w[1]*x[2] + w[2]*x[4] 

This can be very useful in some settings to use in conjunction with 0-dilated filters because it allows you to merge spatial information across the inputs much more agressively with fewer layers. Receptive field would grow much quicker.([cs231](http://cs231n.github.io/convolutional-networks/))

DeepMind’s WaveNet
dilated causal convolution layer: make sure to avoid use the future to predict the past without an explosion in model complexity.  
![](https://jeddy92.github.io/images/ts_conv/WaveNet_causalconv.png)  
![](https://jeddy92.github.io/images/ts_conv/WaveNet_dilatedconv.png)

(To be continued)

Reference
----
[Mishra, Nikhil, et al. "A simple neural attentive meta-learner." (2018).](https://arxiv.org/pdf/1707.03141.pdf)  
[Time Series Forecasting with Convolutional Neural Networks - a Look at WaveNet](https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_conv/)   
[Meta-learning PPT from cs294](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-20.pdf)
