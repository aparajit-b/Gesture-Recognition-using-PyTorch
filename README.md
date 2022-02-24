# Gesture Recognition using PyTorch

## Gesture Recognition from Images using Deep Learning

### Data Description

- The data consists of 300 images each of 4 predefined gestures **play**, **vol-down**, **vol-up**, **stop** performed by a single user. 300 images of plain background not containing any gestures have also been collected under the **none** category

### Problem Statement

- To recognize the gestures that are performed by different users over a real-time stream. Gestures performed by the users will be collected as images, and Deep Learning models will be trained over the collected image data. The model will be tested over a real-time stream to recognize the gestures.

- We have 4 predefined gestures - **play**, **vol-down**, **vol-up**, **stop** and have also captured plain background under **none**.

### Packages Used

- The packages used for the project are provided under **requirements.txt** file. However, some of the prominent pacakages used were

  - **torch** (PyTorch)
  - **cv2** (OpenCV)
  - **torchvision** (Dataloaders and image transformations)
  - **torch.nn** - For creating the neural network

### Solution approach

- **Capturing the Images** (***capture_images.py***):  Using the code, we intend to capture frames of a video stream as images. The webcam of our laptop is accessed, and a rectangular frame is created within which the gestures performed are captured and stored under their respective gesture directories within the train and validation sets. 200 training and 100 validation images of size 224 x 224 are captured for each gesture, along with **none** folder with plain background captured (1500 images in total)
- **Training Deep Learning model over the images** (***train.py***): The images collected are sent in batches of 50 for both the train and validation sets, and appropriate transformations are applied to the images. The images are converted to grayscale before applying the model. A CNN model has been defined and applied over the data over 10 epochs. We would obtain the loss and accuracy over both the sets for each epoch. A training accuracy of 99.7% and a validation set accuracy of 78.2% was obtained. The model object is then pickled and stored as a **.pt** file
- **Testing the model over a real-time stream** (***test.py***): Finally, the stored model will be loaded. A webcam stream with a rectangular frame is opened similar to the image capturing process. The model will recognize the gestures applied on the frame and display the same in the webcam stream window.

### Process Improvement

- Application of pre-trained models **(Transfer Learning)**
- Collecting more images in different backgrounds

### Future Scope

- Application of this over a video player to manipulate a video based on the gestures performed

