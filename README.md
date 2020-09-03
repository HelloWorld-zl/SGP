# SGP

The data files are too large to update, so the program cannot be executed. All codes need path to be set correctly for running.
To run the code directly, use "main.py".

The instruction of files and folders are outlined below.
>"Data" stores all datasets extracted from the original images, with different patch sizes.

>"models" stores codes for building neural networks, including:
>>* Basic_network.py: to build general neural network model.
>>* Dilated_CNN.py: to build dilated CNN model.
>>* Normal_CNN.py: to build normal CNN model.

>"Generate_training_data.py": to capture out patchs around sample pixels.

>"main.py": to test all models used in this project. All models will be trained for 10 times, and stored under the specific path.

>"test_new_image_on_nn.py": to test general neural network model on the testing image, the best one of ten should be assigned before running.

>"test_new_image_on_nn32.py": to test general neural network model with patch size 32 on the testing image, the best one of ten should be assigned before running.

>"test_new_image_on_cnn.py": to test CNN model on the testing image, the best one of ten should be assigned before running.

>"utils.py": to define some functions used by above codes, including loading data and plotting training curve.
