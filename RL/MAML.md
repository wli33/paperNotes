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

Implemetation
----------
```
# source code from the reference link
# a: training data for inner gradient, b: test data for meta gradient
 """ Perform gradient descent for one task in the meta-batch. """
 
 task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
 task_lossa = self.loss_func(task_outputa, labela)

grads = tf.gradients(task_lossa, list(weights.values()))
if FLAGS.stop_grad:
    grads = [tf.stop_gradient(grad) for grad in grads]
gradients = dict(zip(weights.keys(), grads))
    
## update theta_prime
fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))
    
## use updated theta_prime to get the pred y in query set and record loss for a single task
output = self.forward(inputb, fast_weights, reuse=True)
task_outputbs.append(output)
task_lossesb.append(self.loss_func(output, labelb))
    
## continue from the second example to the end
for j in range(num_updates - 1):
    ## the loss for the support set, using SGD to update the inner loop.
    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
    grads = tf.gradients(loss, list(fast_weights.values()))
    if FLAGS.stop_grad:
        grads = [tf.stop_gradient(grad) for grad in grads]
        gradients = dict(zip(fast_weights.keys(), grads))
        fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in     fast_weights.keys()]))
        ## send the updated theta_prime to get the pred the kth y in query set and record loss
        output = self.forward(inputb, fast_weights, reuse=True)
        task_outputbs.append(output)
        task_lossesb.append(self.loss_func(output, labelb))

task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

### for meta learning, get the loss for each shot
self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

### the 1st shot support set loss, 'Pre-update loss'
self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

### use the last shot loss to bp, 'Post-update loss'
self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates-1])                
self.metatrain_op = optimizer.apply_gradients(gvs)

```
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
