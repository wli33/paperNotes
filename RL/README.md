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

(To be continued)

Reference
----
[Mishra, Nikhil, et al. "A simple neural attentive meta-learner." (2018).](https://arxiv.org/pdf/1707.03141.pdf)
