RL2: FAST REINFORCEMENT LEARNING VIA SLOW REINFORCEMENT LEARNING
---------
The learning process of deep RL requires a huge number of trials.   
The authors investigated meta-learning in reinforcement-learning domains using traditional RNN architectures. 

Method
---------
Assume a set of MDPs,and a distribution over them. sample a fixed MDP, and applied to a trial. each trial have n=2 episodes. 
For each episode, a fresh s0 is drawn from the initial state distribution specific to the corresponding MDP. The hidden state is preserved to the next episode, but not preserved between trials.  
Input:(current state, last action, last reward, flagTermination)
The output of the GRU is fed to a fully connected layer followed by a softmax function.
Objective: minimizing the cumulative pseudo-regret (Bubeck & Cesa-Bianchi, 2012):The cumulative regret is the difference between the player’s accumulated reward and the maximum the player could have obtained had she known all the parameters  
![](https://image.slidesharecdn.com/nips2016sharingpart2-170422013135/95/nips-2016-sharing-rl-and-others-33-638.jpg?cb=1492824941)

Optimization: TRPO(Schulman et al., 2015),optionally apply Generalized Advantage Estimation (GAE) (Schulman et al., 2016) to further reduce the variance.  

Evaluation
------------
Two sets of experiments:

MABs and Tabular (to test if the algorithm is close to being optimal)  
Vision-based navigation task (to test scalability)  

(To be continued)  

Reference
----
[Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, and Pieter Abbeel. Rl$ˆ2$:
Fast reinforcement learning via slow reinforcement learning. arXiv preprint arXiv:1611.02779,
2016](https://arxiv.org/pdf/1611.02779.pdf)
