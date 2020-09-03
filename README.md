# SGP

The data files are too large to update, so the program cannot be executed. All codes need path to be set correctly for running.
To run the code directly, use "main.py".

The instruction of files and folders are outlined below.
  "Data" stores all datasets extracted from the original images, with different patch sizes.\<br>  
  
  "models" stores codes for building neural networks, including:\<br>  
    * Basic_network.py: to build general neural network model.\<br>  
    * Dilated_CNN.py: to build dilated CNN model.\<br>  
    * Normal_CNN.py: to build normal CNN model.\<br>  
   
  "Generate_training_data.py": to capture out patchs around sample pixels.\<br>  
  
  "main.py": to test all models used in this project. All models will be trained for 10 times, and stored under the specific path.\<br>  
  
  "test_new_image_on_nn": to test general neural network model on the testing image, the best one of ten should be assigned before running.\<br>  
  
  "test_new_image_on_nn32": to test general neural network model with patch size 32 on the testing image, the best one of ten should be assigned before running.\<br>  
  
  "test_new_image_on_cnn": to test CNN model on the testing image, the best one of ten should be assigned before running.\<br>  
  
  "utils.py": to define some functions used by above codes, including loading data and plotting training curve.\<br>  
