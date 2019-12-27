# ECC_cupy
OpenCV ECC alignment re-implemented by cupy. The original implement is in [OpenCV](https://github.com/opencv/opencv).  (没错，就是抄人家代码).  The results are slightly different from OpenCV implement. And it's only sometimes faster than the CPU version. T_T

Tested on python 3.6 & cupy-cuda92

###### Requirements:

​	cupy: please refer to https://cupy.chainer.org/

​	numpy

###### Usage:

​	Same to cv2.findtransformECC, the type returned is numpy.ndarray.

```python
coo, mat = cp_findtransformECC(template_image, input_image, warp_matrix=None, motion_type=MOTION_EUCLIDEAN, criteria=None,
                        input_mask=None, gauss_filter_size=5)
​```
```




###### Reference:

@article{article,
author = {Evangelidis, Georgios and Psarakis, Emmanouil},
year = {2008},
month = {11},
pages = {1858-65},
title = {Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization},
volume = {30},
journal = {IEEE transactions on pattern analysis and machine intelligence},
doi = {10.1109/TPAMI.2008.113}
}