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
![](https://image.slidesharecdn.com/nips2016sharingpart2-170422013135/95/nips-2016-sharing-rl-and-others-33-638.jpg?cb=1492824941)

(To be continued)  

Reference
----
[Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, and Pieter Abbeel. Rl$ˆ2$:
Fast reinforcement learning via slow reinforcement learning. arXiv preprint arXiv:1611.02779,
2016](https://arxiv.org/pdf/1611.02779.pdf)
