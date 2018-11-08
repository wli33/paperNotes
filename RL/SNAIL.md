A SIMPLE NEURAL ATTENTIVE META-LEARNER
---------
Deep neural networks tend to struggle:   
1. when data is scarce;  
2. when need to adapt quickly to changes in the task.  
Solution: training a meta-learner on a distribution of similar tasks, hope to generalize to novel but related
tasks by learning a high-level strategy that captures the essence of the problem it is asked to solve.

Many recent meta-learning approaches are extensively hand-designed:   
architectures specialized to a particular application, or hard-coding algorithmic components that constrain how the meta-learner solves the task. 

The author propose Simple Neural AttentIve Learner (or SNAIL) that use a combination of convolutions and soft attention layers. 

![](https://lilianweng.github.io/lil-log/assets/images/snail.png)
In reinforcement-learning settings, it receives a sequence of observation-action-reward tuples.

###Experiments
1. Image classification.  
Dataset: 5-way Omniglot, 20-way Omniglot, and 5-way mini-ImageNet. 
sample N classes from the dataset and K examples of each class, followed by a new, unlabeled example from one of the N classes.  
report the average accuracy on this last, (NK + 1)-th timestep. 

2. Reinforcement learning(MULTI-ARMED BANDITS,TABULAR MDPS,CONTINUOUS CONTROL, visual navigation).
Learning to visually navigate a maze  
- train on 1000 small mazes  
- test on held-out small mazes and large mazes   
![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/7e9c1e0d247b20a0683f4797d9ea248c3b53d424/13-Figure5-1.png)  
Results: fast time to find goal.

(To be continued)

Reference
----
[Mishra, Nikhil, et al. "A simple neural attentive meta-learner." (2018).](https://arxiv.org/pdf/1707.03141.pdf)
