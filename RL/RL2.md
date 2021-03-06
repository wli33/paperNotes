RL2: FAST REINFORCEMENT LEARNING VIA SLOW REINFORCEMENT LEARNING
---------
The learning process of deep RL requires a huge number of trials.   
The authors investigated meta-learning in reinforcement-learning domains using traditional RNN architectures. 

Method
---------
Assume a set of MDPs,and a distribution over them. sample a fixed MDP, and applied to a trial. each trial have e.g. n=2 episodes. 
For each episode, a fresh s0 is drawn from the initial state distribution specific to the corresponding MDP. The hidden state is preserved to the next episode, but not preserved between trials.  
* Input:(current state, last action, last reward, flagTermination)
* The output of the GRU is fed to a FC layer followed by a softmax function.
* Objective: minimizing the cumulative pseudo-regret (Bubeck & Cesa-Bianchi, 2012):The cumulative regret is the difference between the player’s accumulated reward and the maximum the player could have obtained had she known all the parameters  
![](https://image.slidesharecdn.com/nips2016sharingpart2-170422013135/95/nips-2016-sharing-rl-and-others-33-638.jpg?cb=1492824941)

Optimization: TRPO(Schulman et al., 2015),optionally apply Generalized Advantage Estimation (GAE) (Schulman et al., 2016) to further reduce the variance.  

Evaluation
------------
Two sets of experiments:

* MABs and Tabular MDPs (to test if the algorithm is close to being optimal)  
MABs compared with: Ramdom, Gittins index, UCB1,Thompson sampling,e-Greedy, Greedy.
Tabular MDPs compared with: Ramdom(uniform action), PSRL(generalization of Thompson sampling), BEB(adds an exploration bonus),UCRL2,e-Greedy, Greedy.
* Vision-based navigation task (to test scalability)  
receives +1 reward when reaches the target, −0.001 when it hits the wall, and −0.04 per time step to encourage it to reach targets faster.
Visual navigation alone is challenging:
1. very sparse rewards during training, and does not have the primitives for efficient exploration at the beginning of training. 
2. needs to make efficient use of memory to decide how it should explore the space, without forgetting about where it has already explored.  

[Figure 6](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/051d94a2aec708fe353319ab4a9643b66daa6e89/8-Figure6-1.png)

6a and 6b: the agent should remember the target’s location, and utilize it to act optimally in the second episode.
6c and 6d: occasionally the agent forgets about where the target was, and continues to explore in the second episode.(Need better RL techniques used as the outer-loop algorithm) 


Reference
----
[Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, and Pieter Abbeel. Rl$ˆ2$:
Fast reinforcement learning via slow reinforcement learning. arXiv preprint arXiv:1611.02779,
2016](https://arxiv.org/pdf/1611.02779.pdf)
