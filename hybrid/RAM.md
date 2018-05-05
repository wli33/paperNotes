An attention-based model may be better at both dealing with clutter and scaling up to large input images.
![](https://raw.githubusercontent.com/torch/torch.github.io/master/blog/_posts/images/rva-diagram.png)   
A. Retina and location encoding:
the first patch being gw × gw pixels, and each successive patch having twice the width of the previous. The k patches are then all resized to gw × gw and concatenated. Glimpse locations l were encoded as real-valued (x, y) coordinates2 with (0, 0) being the center of the image x and (−1, −1) being the top left corner of x.
```
def get_glimpse(self, loc):
    """
    Take glimpse on the original images.
    loc: size of [batch_size,2], containing the y, x locations of the center of each window.
    k=1 patch
    """
    imgs = tf.reshape(self.images_ph, [
        tf.shape(self.images_ph)[0], self.original_size, self.original_size,
        self.num_channels
    ])
    glimpse_imgs = tf.image.extract_glimpse(imgs,
                                            [self.win_size, self.win_size], loc)
    glimpse_imgs = tf.reshape(glimpse_imgs, [
        tf.shape(loc)[0], self.win_size * self.win_size * self.num_channels
    ])
    return glimpse_imgs
```
B. Glimpse network:  fg(x, l) had two fully connected layers.  g = Rect(Linear(hg) + Linear(hl)) where hg = Rect(Linear(ρ(x, l)))
and hl = Rect(Linear(l)).  The dimensionality of hg and hl was 128 while the dimensionality of g was 256.
```
def __call__(self, loc):
    """
    self.b_g0 = 128
    self.b_g1 = 256
    """
    glimpse_input = self.get_glimpse(loc)
    glimpse_input = tf.reshape(glimpse_input,
                               (tf.shape(loc)[0], self.sensor_size))
    g = tf.nn.relu(tf.nn.xw_plus_b(glimpse_input, self.w_g0, self.b_g0))
    g = tf.nn.xw_plus_b(g, self.w_g1, self.b_g1)
    l = tf.nn.relu(tf.nn.xw_plus_b(loc, self.w_l0, self.b_l0))
    l = tf.nn.xw_plus_b(l, self.w_l1, self.b_l1)
    g = tf.nn.relu(g + l)
    return g
```
C. Location network: outputs the mean of the location policy at time, defined as fl(h) = Linear(h). it's a two-component Gaussian with a
fixed variance.
```
def __call__(self, input):
    # fl(h) = Linear(h) is the mean of the location. from 256 cell size to 2(y,x) 
    # loc is discrete value hence non-differentiable. need to stop gradient to stop flow backward to loss
    mean = tf.clip_by_value(tf.nn.xw_plus_b(input, self.w, self.b), -1., 1.)
    mean = tf.stop_gradient(mean)
    if self._sampling:
      loc = mean + tf.random_normal(
          (tf.shape(input)[0], self.loc_dim), stddev=self.loc_std)
      loc = tf.clip_by_value(loc, -1., 1.)
    else:
      loc = mean
    loc = tf.stop_gradient(loc)
    return loc, mean
```
D. Core network: LSTM.  ht = fh(ht−1) = Rect(Linear(ht−1) + Linear(gt)). h0:zero_state, g1:random.
```
# number of examples
N = tf.shape(images_ph)[0]
init_loc = tf.random_uniform((N, 2), minval=-1, maxval=1)
init_glimpse = gl(init_loc)
# Core network.
lstm_cell = rnn_cell.LSTMCell(config.cell_size, state_is_tuple=True)
init_state = lstm_cell.zero_state(N, tf.float32)
inputs = [init_glimpse]
inputs.extend([0] * (config.num_glimpses)) # 8 steps
outputs, _ = seq2seq.rnn_decoder(
    inputs, init_state, lstm_cell, loop_function=get_next_input)
    
def get_next_input(output, i):
  loc, loc_mean = loc_net(output)
  gl_next = gl(loc)
  loc_mean_arr.append(loc_mean)
  sampled_loc_arr.append(loc)
  return gl_next
```
perform a softmax classification to the last hidden state:
```
output = outputs[-1]
# Build classification network.
with tf.variable_scope('cls'):
  w_logit = weight_variable((config.cell_output_size, config.num_classes))
  b_logit = bias_variable((config.num_classes,))
logits = tf.nn.xw_plus_b(output, w_logit, b_logit)
#softmax = tf.nn.softmax(logits)

# cross-entropy.
xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_ph)
xent = tf.reduce_mean(xent)
pred_labels = tf.argmax(logits, 1)
```
(To be continued)
### Reference
[Recurrent Models of Visual Attention V. Mnih et al](https://arxiv.org/pdf/1406.6247.pdf)   
[codes](https://github.com/zhongwen/RAM)
