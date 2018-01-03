While reading the paper [Fully Convolutional Networks
for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf), some steps are ambiguous to me, e.g. section 4.3 Combining what and where:"our skips are implemented by first scoring each layer to be fused by 1 × 1 convolution, carrying out any necessary interpolation and alignment, and then summing the scores."

![](https://i.stack.imgur.com/1IPxQ.png)

In their [implemention](https://github.com/shelhamer/fcn.berkeleyvision.org), take fcn16 as an example, the authors use caffe to get  a trans-conv layer and crop the pool4 to the same size, then the fuse layer adds element-wise.
```
n.upscore2 = L.Deconvolution(n.score_fr,
        convolution_param=dict(num_output=21, kernel_size=4, stride=2,
            bias_term=False),
        param=[dict(lr_mult=0)])

n.score_pool4 = L.Convolution(n.pool4, num_output=21, kernel_size=1, pad=0,
      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
n.score_pool4c = crop(n.score_pool4, n.upscore2)
n.fuse_pool4 = L.Eltwise(n.upscore2, n.score_pool4c,
            operation=P.Eltwise.SUM)
```
Then they recover to the original size and do a prediction.
```
n.upscore16 = L.Deconvolution(n.fuse_pool4,
        convolution_param=dict(num_output=21, kernel_size=32, stride=16,
            bias_term=False),
        param=[dict(lr_mult=0)])

 n.score = crop(n.upscore16, n.data)
 n.loss = L.SoftmaxWithLoss(n.score, n.label,
            loss_param=dict(normalize=False, ignore_label=255))
 ```

since I am not Caffe user, I also read a [tensorflow implemetion](https://github.com/sagieppel/Fully-convolutional-neural-network-FCN-for-semantic-segmentation-Tensorflow-implementation/blob/master/BuildNetVgg16.py) as a reference to usderstand:

```
#now to upscale to actual image size
deconv_shape1 = self.pool4.get_shape()  # Set the output shape for the the transpose convolution output take only the depth since the transpose convolution will have to have the same depth for output
W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_CLASSES],name="W_t1")  # Deconvolution/transpose in size 4X4 note that the output shape is of  depth NUM_OF_CLASSES this is not necessary in will need to be fixed if you only have 2 catagories
b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
self.conv_t1 = utils.conv2d_transpose_strided(self.conv8, W_t1, b_t1, output_shape=tf.shape(self.pool4))  # Use strided convolution to double layer size (depth is the depth of pool4 for the later element wise addition
self.fuse_1 = tf.add(self.conv_t1, self.pool4, name="fuse_1")  # Add element wise the pool layer from the decoder

deconv_shape2 = self.pool3.get_shape()
W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
self.conv_t2 = utils.conv2d_transpose_strided(self.fuse_1, W_t2, b_t2, output_shape=tf.shape(self.pool3))
self.fuse_2 = tf.add(self.conv_t2, self.pool3, name="fuse_2")

shape = tf.shape(rgb)
W_t3 = utils.weight_variable([16, 16, NUM_CLASSES, deconv_shape2[3].value], name="W_t3")
b_t3 = utils.bias_variable([NUM_CLASSES], name="b_t3")

self.Prob = utils.conv2d_transpose_strided(self.fuse_2, W_t3, b_t3, output_shape=[shape[0], shape[1], shape[2], NUM_CLASSES], stride=8)

#--------------------Transform  probability vectors to label maps-----------------------------------------------------------------
self.Pred = tf.argmax(self.Prob, dimension=3, name="Pred")
```
