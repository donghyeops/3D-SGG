import matplotlib.pyplot as plt
import os
import os.path as osp
import glob
import cv2

image_path = '/media/ailab/D/ai2thor/roi_images'

file_paths = glob.glob(f'{image_path}/*.jpg', recursive=True)
file_paths.sort()

areas = []

for i, file_path in enumerate(file_paths):
    image = cv2.imread(file_path)
    #print(image.shape)
    area = image.shape[0] * image.shape[1]
    areas.append(area)
    #break
areas.sort()

def make_histogram(plot, data, title='total'):
    plot.hist(x=data, bins='auto', color='#0504aa', rwidth=0.85, alpha=0.85)
    plot.grid(axis='y', alpha=0.75)
    plot.set_xlabel(f'Area [#: {len(data)}]')
    plot.set_ylabel('Frequency')
    plot.set_title(title)
    
sub1 = plt.subplot(211)
sub2 = plt.subplot(223)
sub3 = plt.subplot(224)
plt.tight_layout()

make_histogram(sub1, areas, title='1.0')
make_histogram(sub2, areas[:int(len(areas)*0.4)], title='0.4')
make_histogram(sub3, areas[:int(len(areas)*0.2)], title='0.2')
plt.show()