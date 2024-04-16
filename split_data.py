import os
from sklearn.model_selection import train_test_split
import shutil

# Directory containing your files
source_directory = 'x-ray_Resized_256'
destination_directory = 'split/xray'

# Get all files in the directory
files = os.listdir(source_directory)

# Split in train, test, valm
X,y = train_test_split(files, test_size=0.4)
val, test = train_test_split(y, test_size=0.5)

# create directories 
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)
    os.makedirs(destination_directory+'/train')
    os.makedirs(destination_directory+'/test')
    os.makedirs(destination_directory+'/val')

# build train folder 
for file_name in X:
    source_file_path = os.path.join(source_directory, file_name)
    destination_file_path = os.path.join(destination_directory+'/train', file_name)
    shutil.copy2(source_file_path, destination_file_path)

# build val folder 
for file_name in val:
    source_file_path = os.path.join(source_directory, file_name)
    destination_file_path = os.path.join(destination_directory+'/val', file_name)
    shutil.copy2(source_file_path, destination_file_path)

# build test folder 
for file_name in test:
    source_file_path = os.path.join(source_directory, file_name)
    destination_file_path = os.path.join(destination_directory+'/test',file_name)
    shutil.copy2(source_file_path, destination_file_path)

