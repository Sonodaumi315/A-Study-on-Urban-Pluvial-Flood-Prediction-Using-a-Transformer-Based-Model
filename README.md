# A-Study-on-Urban-Pluvial-Flood-Prediction-Using-a-Transformer-Based-Model
Data, codes, and model parameters in the study of Predicting Urban Pluvial Flood Water Depth Using a Transformer-Based Deep Learning Model with Data Augmentation (Since our research has not yet been formally published, only the code related to the model structure is publicly available)

This is the instruction for utilizing the codes and data from the study of **Predicting Urban Pluvial Flood Water Depth Using a Transformer-Based Deep Learning Model with Data Augmentation**

1. Install python dependencies by running:  
pip install -r requirements.txt

2. Download the original data used in the study by Löwe et al. (2021) from https://data.dtu.dk/articles/software/U-FLOOD_-_computer_code_and_data_associated_with_the_article_U-FLOOD_topographic_deep_learning_for_predicting_urban_pluvial_flood_water_depth_/14206838/1   
After unzip the file, please run "A_LoadData.py" to get 2 datafile, "X_data.npz" and "wsheds.npz".  
Then, copy "X_data.npz", "wsheds.npz", and "Datafiles" into the folder "data" .

3. Download the published trained model parameters (will be published as soon as possible) of MobileViTv2 (or ResU-Net) from Releases for your test (Please don't download the outdated code in the releases). Copy the model you downloaded into the folder "models".

4. "eval.py" under "code" folder can be employed for model evaluation.
In the main function in "eval.py", you can modify the model class, model path, and evaluation function to reproduce the certain results in our study (will be published as soon as possible).
 
5. If you want to trained your own model, you can run "train.py" and modify the training hyper-parameters.
The meaning of hyper-parameters are annotated at the beginning of "train.py" when these variables are defined.  
In the main function in "train.py", you can modify the hyper-parameters for training.
The saved models during training are under "models/_start time of training_" folder.  
Our models were distributed trained using 4 Nvidia Tesla V100.
