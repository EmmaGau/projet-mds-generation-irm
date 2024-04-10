import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

img_folder = 'data/Results/Split3/TEST/IMG'
mask_folder = 'data/Results/Split3/TEST/MASK'
gt_folder = 'data/Results/Split3/TEST/GT'

file_names = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]

img_to_show = [0,1,3]

plt.figure(figsize=(12, 12))

for i in range(len(img_to_show)):
    plt.subplot(3, 3, 3*i+1)
    plt.axis('off')
    plt.imshow(cv2.imread(os.path.join(img_folder, file_names[img_to_show[i]])))
    if i == 0:
        plt.title('Original image', fontsize=20)

    plt.subplot(3, 3, 3*i+2)
    plt.axis('off')
    plt.imshow(cv2.resize(cv2.imread(os.path.join(gt_folder, file_names[img_to_show[i]])), (64,64)))
    if i == 0:
        plt.title('Ground truth mask', fontsize=20)

    plt.subplot(3, 3, 3*i+3)
    plt.axis('off')
    plt.imshow(cv2.imread(os.path.join(mask_folder, file_names[img_to_show[i]])))
    if i == 0:
        plt.title('Generated mask', fontsize=20)

# Add legend
categories = {
'Enamel' : ([255,0,0], 1),
'Bone' : ([0,255,0], 2),
'Dentine': ([0,0,255], 3),
'Other' : ([255, 0, 254], 4),
'Cavity': ([255,152,0], 5),
'Pulp': ([0, 255, 237], 6)
}

handles = [plt.Rectangle((0,0),1,1, color=np.array(categories[label][0])/255) for label in list(categories.keys())]
plt.legend(handles, list(categories.keys()), loc="lower right", bbox_to_anchor=(0.5, -0.5), ncol=3, fontsize=20)

# Remove axis and frame
# plt.tight_layout()
plt.savefig('data/Results/Split3/TEST/compare.png', bbox_inches='tight', pad_inches=0)

plt.show()