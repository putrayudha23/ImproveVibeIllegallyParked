import os
import cv2
from sklearn.svm import LinearSVC
from skimage import feature
import pickle

# Get the HOG Features from the Training Images 
images = []
labels = []
# get all the image folder paths
image_paths = os.listdir(f'./input_hog/train')
for path in image_paths:
    # get all the image names
    all_images = os.listdir(f"./input_hog/train/{path}")
    # iterate over the image names, get the label
    for image in all_images:
        image_path = f"./input_hog/train/{path}/{image}"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # for grayscale image
        image = cv2.resize(image, (144,144)) # input image size = 144x144 pixel
        
        # get the HOG descriptor for the image
        hog_desc = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    
        # update the data and labels
        images.append(hog_desc)
        labels.append(path)

# train Linear SVC
print('Training on train images...')
svm_model = LinearSVC(random_state=42, tol=1e-5)
svm_model.fit(images, labels)

# save training model
with open('modelSVM_Gray144.pkl','wb') as f:
    pickle.dump(svm_model,f)

