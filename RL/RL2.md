RL2: FAST REINFORCEMENT LEARNING VIA SLOW REINFORCEMENT LEARNING
---------
The learning process of deep RL requires a huge number of trials.   
The authors investigated meta-learning in reinforcement-learning domains using traditional RNN architectures. 

Method
---------
Assume a set of MDPs,and a distribution over them. sample for a fixed MDP, and applied to a trial.
each trial have n=2 episodes. 
For each episode, a fresh s0 is drawn from the initial state distribution specific to the corresponding MDP. The hidden state is preserved to the next episode, but not preserved between trials.  
Input:(current state, last action, last reward, flagTermination)
![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/051d94a2aec708fe353319ab4a9643b66daa6e89/2-Figure1-1.png)

(To be continued)  

Reference
----
[Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, and Pieter Abbeel. Rl$ˆ2$:
Fast reinforcement learning via slow reinforcement learning. arXiv preprint arXiv:1611.02779,
2016](https://arxiv.org/pdf/1611.02779.pdf)
