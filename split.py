import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Set the path to the directory containing the images
data_dir = "paste your image folder path here"

# Set the proportion of images to be used for testing
test_size = 0.2

# Get a list of all the image file names in the directory
image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# Split the image file names into training and testing sets
train_files, test_files = train_test_split(image_files, test_size=test_size)

# Create directories for the train and test sets with correct permissions
train_dir = os.path.join(data_dir, "train")
if not os.path.exists(train_dir):
    os.makedirs(train_dir, mode=0o755)

test_dir = os.path.join(data_dir, "test")
if not os.path.exists(test_dir):
    os.makedirs(test_dir, mode=0o755)

# Copy the training images to the train directory with correct permissions
for file in train_files:
    try:
        shutil.copy(file, os.path.join(train_dir, os.path.basename(file)))
        os.chmod(os.path.join(train_dir, os.path.basename(file)), 0o644)
    except IOError as e:
        print(f"Unable to copy file. {e}")

# Copy the testing images to the test directory with correct permissions
for file in test_files:
    try:
        shutil.copy(file, os.path.join(test_dir, os.path.basename(file)))
        os.chmod(os.path.join(test_dir, os.path.basename(file)), 0o644)
    except IOError as e:
        print(f"Unable to copy file. {e}")
