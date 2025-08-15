# Project_2_NLP
In this proeject we were tasked with predicting Fake News articles. 
We trained models on a prelabled data set and predicted on a unlabled dataset. In the end we checked our accuracy.

Our best results:
With Logistic Regression and TimeSeriesSplit.
Accuracy on our prediction against the real labels: 0.9864
F1 Score: 0.975
Precision: 0.996		
Recall: 0.990

Documents in repo:
accuracy_calculation.ipynb.   -I used to calculate the predictions against the real labels
Model_training.ipynb   -Our notebook with our model training results
model_training.py   -Py file of our first basic models
model:training_final.py   -py file of our final models
modelLR.pkl   -Our best preforming model.
NLP Project 2 - Presentation Model.pptx.   -short presentation of our model results
predict.ipynb   -File we used to make the predictions on unlabeled dataset
predict_trans_googlecollab.ipynb   -File we used to make prediction on unlabeled dataset with our model including distilbert.
predictionsLR.csv, predictionsRF.csv, predictionsLR_tranformers_with_original_data.csv   -Our prediction files.
Project_nlp_transformer.ipynb   -Notebook were we tried making model using pretrained distilbert. 

By: Group 2, Alex and Daan
