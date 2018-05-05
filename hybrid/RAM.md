An attention-based model may be better at both dealing with clutter and scaling up to large input images.
![](https://raw.githubusercontent.com/torch/torch.github.io/master/blog/_posts/images/rva-diagram.png)   
A. Retina and location encoding:
the first patch being gw × gw pixels, and each successive patch having twice the width of the previous. The k patches are then all resized to gw × gw and concatenated. Glimpse locations l were encoded as real-valued (x, y) coordinates2 with (0, 0) being the center of the image x and (−1, −1) being the top left corner of x.
```
def get_glimpse(self, loc):
    """Take glimpse on the original images.
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
(To be continued)
### Reference
[Recurrent Models of Visual Attention V. Mnih et al](https://arxiv.org/pdf/1406.6247.pdf)   
[codes](https://github.com/zhongwen/RAM)
