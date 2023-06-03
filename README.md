CAS: Breast Cancer Diagnosis Framework Based on Lesion Region Recognition in Ultrasound Images.


Requirements
Pycharm.Keras and TensorFlow

Hardware environment 
11th Gen Intel(R) Core(TM) i7-11700K CPU @ 3.60GHz and an NVIDIA GeForce RTX 3060 graphics card, running on a 64-bit Windows operating system.

For all experiments,we split the dataset into a training set and a test set in an 80:20 ratio and used the Adam optimizer for parameter updates.

Segmenting tumor lesions
For the segmentation experiment, we use dice_loss as the loss function. The initial learning rate is set to 6e-4, the epoch is set to 60, and the batch size is set to 12. 

classifying tumor benignity or malignancy
For the classification experiment, we use sparse_categorical_crossentropy as the loss function. The learning rate is set to 0.001, the epoch to 55, and the batch size to 16.
