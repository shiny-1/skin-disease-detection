# Skin-disease-detection
This is a flask application. This skin disease detetction app is designed to recognise the type of skin disease through image processing techniques and deep learning algorithms. This app is designed to identify 8 different types od skin diseases. Along with the detection of skin disease type it also provide i nformation regarding the disease symptoms precautions and medication details.In this app the disesae is identified by uploading an skin image and if the user does not upload an image of skin then app shows that it is not a skin image by using skin detection code and also it specifies if the disesae is not among the 8 different types of diseases and also warns about the quality of image if a low quality image is given

 ## Tech stack used
 The frontend is developed by using html, css. javascript
 The backend is developed by using Falsk framework
 The deep learning model is developed by using Efficientnet

 ## How the files are stored
 ### flask Folder-> The frontend and the main flaask application(skin-app) is stored in this folder. the application gets executed if you run the skinapp python source file
 ### Main-code-> This foldder contains the deeplearning model and the h5 file which contains the best weights of efficient model which is giving the best accuracy in detecting 8 different types of diseases 

## Transfer Learning
 Transfer learning is a machine learning technique where a model trained on one task 
is re- purposed or adapted for a related task. Instead of training a model from scratch, 
transfer learning leverages the knowledge gained from solving one problem and applies it to
a different but related problem

Pre-trained Model:
Start with a model that has been trained on a large dataset for a specific task, such as image
classification using a Convolutional Neural Network (CNN) trained on ImageNet.

Feature Extraction:
pre-trained model's parameters are used as a feature extractor. The layers close to the input 
are frozen, meaning their weights are not updated during training.


Fine-Tuning:
Optionally, you can unfreeze some of the top layers of the pre-trained model and train them
along withthe new task-specific layers. This step allows the model to adapt to the nuances of 
the new task.


Training New Data:
Train the adapted model on the new dataset specific to the task you want tosolve. This
dataset is usually smaller than the original dataset used to train the pre-trained model


# Efficient Net
 Is a family of convolutional neural network architectures that are designed to 
achieve state- of-the-art performance with significantly fewer parameters and computational 
resources compared to traditional CNN architectures. The theory behind EfficientNet 
revolves around improving the efficiency of neural networks by balancing model depth, 
width, and resolution using a compound scaling method. Here's a detailed overview of the
theory behind EfficientNet
