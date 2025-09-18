To train a face recognition system, we need two main elements:
1) Face features
2) Labels or identifiers of people

For this purpose, we define two empty lists, features and labels. Now, instead of storing people's names in the labels list, we store their name indices in the people list, because we prefer to work with numbers for training the model. 

In the next step, a pre-trained model (Haar Cascade) is loaded, which we use to detect faces in images. The haar_face.xml file should be downloaded and available in the project folder path. To access it, this link is used in OpenCV.
https://github.com/opencv/opencv/tree/master/data/haarcascades

The create_train function is created, which is responsible for reading images, identifying faces, extracting features, and labeling. First, it assigns a numeric label to each person in the people list using the enumerate loop. Then, it specifies the folder path of each person and reads all the images in that folder. For each image, the image is first loaded with cv.imread and then each image is converted to grayscale to make the face recognition process faster and more accurate. In the next step, we identify faces using haar_cascade and use detectMultiScale to return the coordinates of the rectangle around the faces. Now, instead of drawing a rectangle around the faces, we extract only the face region or ROI and store it as a feature in the feature list. . This feature is stored as an array of type uint8 to be compatible with the training algorithm and is finally placed in the features list.
Finally, for each image, if a face was detected, the face region was extracted, and the corresponding features and labels were stored. Finally, by executing this function, the features and labels lists are built from all the training images and are ready to train the model.

For training, the Face Recognizer available in OpenCV was used. In this project, we used the LBPHFaceRecognizer model, which performs well in different lighting conditions. First, the model was built and then trained on the training data using the .train method.

To test the model, we saved the features and labels using np.save and also saved the trained model in the .yml format.

Now we train the model. The equality of the number of features and labels indicates that the model training was done correctly.

In the testing phase of the face recognition model, first the Haar classifier for face recognition and the trained LBPH model, which was previously trained with the training images, are loaded. Then, the names of people (labels) are extracted from the training folder so that the numerical labels in the model output can be converted to people's names. Next, using the images in the validation folder, each image is loaded individually and converted to grayscale, and then the face region in it is identified using the Haar classifier. If a face is found in the image, that region is given to the face recognition model to make a prediction, and the model output includes a numeric label (along with a confidence level) that is converted to the corresponding person name. If this name is the same as the actual name in the person folder, the prediction is considered correct and counted. Finally, after examining all the images, the number of correct predictions is calculated over the total images and reported as the final accuracy of the model, which indicates how well the trained model performs in recognizing faces in the unseen data.

First, the Haar classifier is loaded from the haar_face.xml file. This classifier is used to identify the location of the face in the image. A face recognizer object is created based on the LBPH algorithm. The previously saved trained model is then loaded with read() . This model contains the extracted features and the labels of the people. The list of people folders in the training path is read and sorted with sorted() to match the order of the labels stored in the training file. The list of people is then printed.

In the next step, the image is converted to grayscale, since face recognition works best on grayscale images. Then, the location of the face in the image is detected using detectMultiScale . This function returns rectangles for the faces. The total count of the images is incremented by one, and the variable recognized indicates whether a face was successfully recognized in this image or not.

Next, for each recognized face, a face region (ROI) is extracted from the image. This region is resized to 200×200. This step is important for smoothing the model input. Then, the prediction is made using the LBPH model. The name of the person is extracted from the people list using label and the prediction result is printed for each image; if the predicted name matches the actual name, the count of correct predictions is incremented.

In the face recognition process using the LBPHFaceRecognizer model in the OpenCV library, the confidence value indicates the level of uncertainty of the model regarding the prediction made. Unlike many machine learning models where a higher confidence value means more confidence, in the LBPH model the lower the confidence value, the more confident the model is in its prediction.
One way to improve the accuracy of the model is to use a specific threshold for the confidence value. Only predictions whose confidence value is less than a certain threshold (for example, 80) are considered valid predictions. This filters out predictions with high uncertainty and increases the accuracy and realism of model performance evaluation.
