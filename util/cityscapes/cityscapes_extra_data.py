# Extra information about nyuv2
import numpy as np

# Number of classes:
n_classes = 19

# set 0 to first label
# weights=np.ones(1, n_classes)

weights=np.ones(n_classes)

colors = [#[  0,   0,   0],
            [128,  64, 128],
            [244,  35, 232],
            [ 70,  70,  70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170,  30],
            [220, 220,   0],
            [107, 142,  35],
            [152, 251, 152],
            [ 0, 130, 180],
            [220,  20,  60],
            [255,   0,   0],
            [  0,   0, 142],
            [  0,   0,  70],
            [  0,  60, 100],
            [  0,  80, 100],
            [  0,   0, 230],
            [119,  11,  32]]

label_colours = dict(zip(range(19), colors))