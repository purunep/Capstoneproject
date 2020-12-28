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
For the experiment, we have used the different parameters for  **automl** settings as below:

| Parameter                   | Value                  | Reason                                                                                 |
| ----------------------------|:-----------------------|                                                      ---------------------------------:|
|enable_early_stopping        |True                    | To enable early termination if the score is not improving in the short term            |
|iteration_timeout_minutes    |5                       | To set the maximum time in minutes that each iteration can run for before it terminates| 
|max_concurrent_iterations    |4                       | To specify the maximum number of iterations that would be executed in parallel. | 
|max_cores_per_iteration      |-1                      | To specify the maximum number of threads to use for a given training iteration. -1 means to use all the possible cores per iteration per child-run.      | 
|featurization                |auto                    | To specify wherether featurization should be done automically or not, auto is ued to do it automatically.| 

For the configuration we have used the following parameters: 
| Parameter                   | Value                  | Reason                                                                                 |
| ----------------------------|:-----------------------|                                                      ---------------------------------:|
|experiment_timeout_minutes   |30                    | To specify how long in minutes, our experiment should continue to run         |
|task                         |classification                     | We are going to solve the classification problem| 
|primary_metric               |accuracy                       | The metric that Automated Machine Learning will optimize for model selection. We are going to optimize the Accuracy.| 
|enable_onnx_compatible_models|True                    | To enable ONNX-compatible models.      | 
|compute_target               |cpu_cluster                    | To run teh Automated Machine learning experiment, we are going to use remote created compute cluster | 
|training_data                |train_data                    | To specify wherether featurization should be done automically or not, auto is ued to do it automatically.| 
|label_column_name            |label                    | This is the model value to predict, our lable column is 'Outcome'.| 
|path                         |project_folder                    | The full path to the Azure Machine Learning project folder.| 
|n_cross_validations          |5                    | How many cross validations to perform when user validation data is not specified.| 
|debug_log                    |automl_errors.log                    | The log file to write debug information| 



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
