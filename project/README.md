# Capstone Project

This project is part of the Udacity Azure ML Nanodegree program. In this project, we have used the external data source to predict. We have used Microsoft Azure ML SDK to build the model using AutoML and Hyperparameter tuning. After training the model, we registered the model and deployed with  the SDK. After its deployed, we have used the REST API to predict.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.
Dowload the file from: https://www.kaggle.com/uciml/pima-indians-diabetes-database?select=diabetes.csv
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets?view=azure-ml-py


## Dataset

### Overview
In this project, we are using the data from Kaggle. Here is the [link](https://www.kaggle.com/uciml/pima-indians-diabetes-database?select=diabetes.csv) for the dataset.
There are total 9 columns and 768 entries. 

### Task
The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.
The target (label) is "Outcome" column with values of 1 means has diabetes and 0 means no diabetes. We will be using all independent varialbles from the datasets like: preganicies
the patient has had, their BMI, insulin level, age and so on.

### Access
I have downloaded the data from Kaggle [link](https://www.kaggle.com/uciml/pima-indians-diabetes-database?select=diabetes.csv) and uploaded to my github [link](https://raw.githubusercontent.com/purunep/Capstoneproject/main/project/data/diabetes.csv). We can load the dataset in the Notebook by providing the raw url of the dataset.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
