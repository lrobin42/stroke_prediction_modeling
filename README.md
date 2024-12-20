## Project Premise 

Imagine you've been contracted as a data scientist for Johns Hopkins Hospital. They've asked to you to create a machine learning model, which could 
predict which patients are likely to get a stroke because being able to determine which patients have 
high stroke risk will allow your doctors to advise them and their families on how to act in case of an emergency.

## Project Approach

This project uses the [Stroke Prediction dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle to train and test both single-classifier and ensemble supervised learning models. 
The models are all tree-based models. First, we set a baseline by tuning a single sklearn decision tree classifier, then proceeding to LightGBM gradient boosted tree ensemble models and gradient boosted random forests. 
We then proceed to train XGBoost tree ensembles and random forest classifiers as well.

Once we've calibrated and tested the first string of models, we refine and retrain them by using both under and oversampling to improve model performance and discuss both feature importance and how well the model performs against established knowledge.


## Project Findings
After testing 10 different classifiers, we selected the LightGBM random forest classifier as our final model to give to the client. 

## Recommendations for further research
To strengthen both the explainability and explanatory power of our model, we recommend prioritizing increasing the number of datapoints that we can train our model, as well the number of features that could be of use. Part of the utility of features like gender,
age, and smoking_status is that they directly describe behaviors or patient traits that contribute to a patient's stroke risk, whereas features like work_type, residence_type, and ever_married indirectly point to known risk factors. 

Incorporating these risk factors more directly would strengthen the classifiers--instead of residence_type, we could include number of hours a week spent exercising, or data on water pollution or air pollution in their area. Instead of ever_married,
we could include datapoints around socialization such as the number of hours spent on the phone or in person with friends and family. Not only would having clearer predictors along these lines strengthen our model, but their incorporation would help flesh out our SHAP 
values for features further down the list. 

Alternatively, we could also request more data for age groups and gender sub-populations of interest in the event that they are more worried about possible stroke patients that are between 50 and 72, children, or women that are peri-menopausal or post-menopausal
in particular. From there our team could build different models for these particular subpopulations since risk levels change in relation to a woman's pregnancy and menopausal status, and are also different at different stages of life. 

And lastly, simply having more datapoints across all patient subgroups would help us to train effective classifiers without using SMOTE or other imbalanced learning techniques. 

## Relevant Files
Please check out the stroke_prediction_functions.py file to see the helper functions used, and requirements.txt for module/package information. 
