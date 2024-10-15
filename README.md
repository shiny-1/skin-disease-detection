# skin-disease-detection
This is a flask application. This skin disease detetction app is designed to recognise the type of skin disease through image processing techniques and deep learning algorithms. This app is designed to identify 8 different types od skin diseases. Along with the detection of skin disease type it also provide i nformation regarding the disease symptoms precautions and medication details.In this app the disesae is identified by uploading an skin image and if the user does not upload an image of skin then app shows that it is not a skin image by using skin detection code and also it specifies if the disesae is not among the 8 different types of diseases and also warns about the quality of image if a low quality image is given

 ## tech stack used
 The frontend is developed by using html, css. javascript
 The backend is developed by using Falsk framework
 The deep learning model is developed by using Efficientnet

 ## how the file are stored
 ### flask Folder-> the frontend and the main flaask application(skin-app) is stored in this folder. the application gets executed if you run the skinapp python source file
 ### Main-code-> this foldder contains the deeplearning model and the h5 file which contains the best weights of efficient model which is giving the best accuracy in detecting 8 different types of diseases 
