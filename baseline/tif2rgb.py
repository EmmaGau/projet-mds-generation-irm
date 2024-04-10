import cv2
import os

import argparse
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')

args = parser.parse_args()

os.makedirs(args.output, exist_ok = True)

categories = {
'Email' : ([255,0,0], 1),
'Os' : ([0,255,0], 2),
'Dentine': ([0,0,255], 3),
'Autre' : ([255, 0, 254], 4),
'Carie': ([255,152,0], 5),
'Pulpe': ([0, 255, 237], 6)
}

colors = {v[1] : v[0] for k, v in categories.items()}

for f in os.listdir(args.input):
    tif = cv2.imread(os.path.join(args.input, f), -1)
    print(tif)
    res = np.zeros((tif.shape[0], tif.shape[1], 3), dtype=np.uint8)
    for c_number, c_color in colors.items():
        res[tif == c_number] = c_color
    cv2.imwrite(os.path.join(args.output, f), res)
