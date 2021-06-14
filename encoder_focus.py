import numpy as np
from matplotlib import image
from pyts.image import GramianAngularField

data = np.loadtxt('./data_set/ts_data.csv', delimiter=",", skiprows=0)
label = np.loadtxt('./data_set/label_focus.txt')

image_size = 200
save_path = './output/difference_imgs_focus/'

i = 0

for single_data in data:
    single_data = single_data.reshape(1, -1)
    gafs = GramianAngularField(image_size=image_size, method='difference')
    data_gasf = gafs.fit_transform(single_data)

    draw = data_gasf[0]

    filename = save_path + str(int(label[i])) + '/%d.jpg' % i
    image.imsave(filename, draw)

    i += 1
