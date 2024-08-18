# HAND SIGN RECOGNITION

### Problem definition

The Hand Sign recognition System aims to develop a robust machine learning model that can accurately predict hand signs that would be useful for hand sign translation and accessibility for people with hearing and speech disabilities. 
Link to the dataset: https://www.kaggle.com/datasets/ash2703/handsignimages
The Sign Language MNIST dataset is presented here and follows the jpeg image format with labels . The American Sign Language letter database of hand gestures represent a multi-class problem with 24 classes of letters (excluding J and Z which require motion).
There are a total of 27,455 gray-scale images of size 28*28 pixels whose value range between 0-255. Each case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions).
The goal of the project is to create a model that correctly identifies patterns in the dataset and correctly classify them

So one of the images would look like this:
![image](https://github.com/user-attachments/assets/b0e57eb0-b879-4203-bd30-34d644e9ff6d)



### Methodology
The original size of 28x28 was maintained in the dataset and that was what the model was trained on. Pixel values were normalized and converted to grayscale so as to let Pytorch see it as grayscale images with 1 channel instead of 3 channels even if the images were grayscale 

After the transformation, it should look like this
![image](https://github.com/user-attachments/assets/ba098d51-4e01-46de-a598-c4df9ed394a2)

Not exactly grayscale but the model understands whats going on

The model was developed using a convolutional network of 3 layers with a max pool in only the first layer and batch normalization on all layers and a fully connected layer as the classifier. Other pretrained models were tried but they ended up giving discouraging results so they were discarded for this project.



### Model Evaluation
After 2 epochs, we get a surprising validation accuracy of around 99%. With reasonably low loss scores on both training and validation sets.
It may be surprising but that doesn’t mean that the results are not realistic enough. After making predictions on the entire testing set, we get an accuracy of around 95%. That would mean that it is not entirely perfect but still decently good with regards to this project

These are the predicted values
![predictions](https://github.com/user-attachments/assets/72d737b8-7863-461e-a06c-7baa9ab1f624)


This is the Confusion matrix
![confusion_matrix](https://github.com/user-attachments/assets/d9a71828-3996-4334-b199-639a511ff58c)

### Conclusion
The model was able to do this well because of the nature of the dataset. It has a low pixel size so the model was able to learn patterns quickly and efficiently hence the low epoch rate.
This project just highlights the power of deep neural networks and how they can quickly identify patterns in unstructured data

*Future Improvements could include:*

- Using images with a higher pixel size and resolution like 512x512
- Implementing the model with videos
- Using colored images as opposed to gray scale images



