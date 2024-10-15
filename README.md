# A-Study-on-Urban-Pluvial-Flood-Prediction-Using-a-Transformer-Based-Model
Data, codes, and model parameters in the study of Predicting Urban Pluvial Flood Water Depth Using a Transformer-Based Deep Learning Model with Data Augmentation

This is the instruction for utilizing the codes and data from the study of **Predicting Urban Pluvial Flood Water Depth Using a Transformer-Based Deep Learning Model with Data Augmentation**

1.Install python dependencies by running:  
pip install -r requirements.txt

2.Download the original data used in the study by Löwe et al. (2021) from https://data.dtu.dk/articles/software/U-FLOOD_-_computer_code_and_data_associated_with_the_article_U-FLOOD_topographic_deep_learning_for_predicting_urban_pluvial_flood_water_depth_/14206838/1   
After unzip the file, please run "A_LoadData.py" to get 2 datafile, "X_data.npz" and "wsheds.npz".  
Then, copy "X_data.npz", "wsheds.npz", and "Datafiles" into the folder "data" 
