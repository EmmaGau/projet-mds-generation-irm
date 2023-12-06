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

for f in os.listdir(args.input):
    img = cv2.imread(os.path.join(args.input, f))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tif = np.zeros(img.shape[:2])
    for k, (c, v) in categories.items():
        tif[np.all(img == c, axis=-1)] = v
    cv2.imwrite(os.path.join(args.output, os.path.splitext(f)[0] + '.png'), tif)
