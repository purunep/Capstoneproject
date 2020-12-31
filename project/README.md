# Capstone Project

This project is part of the Udacity Azure ML Nanodegree program. In this project, we have used the external data source to train the model and predict. We have used Microsoft Azure ML SDK to build the model using AutoML and Hyperparameter tuning. After training the model, we registered the best model and deployed as a Web service with the SDK. After its deployed, we have used the REST API and sdk to predict the outcome.

## Azure Auto ML Project Setup
Here are the steps we use to setup our Azure Auto ML:

![alt text](https://github.com/purunep/Capstoneproject/blob/main/project/images/steps.png)


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
Automated machine learning is the process of automating the time consuming , iterative tasks of machine learning model development.
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
The best model we got from the experiment is **VotingEnsemble**. We got an accuracty of 78%. The accuracy chould have been improve by enabling the Deeplearning and 
also increasing the experiment timeout. Here are the screenshot of the **RunDetails**

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
Hyperparameter tuning is the process of finding the configuration of hyperparameters that results in the best performance. We used the LogisticRegression model for the experiment because we need to predict for discrete functions and its easier to implement, interpret and very efficient to train.
For this experiment, we are using **Random sampling** , which supports discrete and continuous hyperparameeters.It supports early termination of low-performance runs.
For the **Random sampling**, we providing parameter **-C** to provide uniform distributed between 0.5 to 1.00. And also using parameter **--max_iter** as choice value of 10, 20 or 30.
For the termination policy, we are using **BanditPolicy**, its based on slack factor and evaluation interval.Bandit terminates runs where the primary metric is not within the 
specified slack factor compared to the best performing run.


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
We got **Accuracy** of : 0.7272 with primary_metric_config goal **maximize**. We used RandomParameterSampling as hyperparameter sampling.
We could improve by implementing different hyperparameter tuning strategies like **Grid Search**.
Here are the run details from the experiment: 

![alt text](https://github.com/purunep/Capstoneproject/blob/main/project/images/hyperdrive_rundetails.png)

## Model Deployment
The best model we got from HyperDrive experiment is of accuracy: 72% whereas the best model we got from Auto ML experiment is of accuracy: 78%.
So, we deployed the model from Auto ML experiment. 
Here are the steps we followed for deploying the model:

After finding the best model, we registered the model by providing the model name. Then we created the deploy configuration and InferenceConfig by providing the
entry script. After that we deployed the web service with ACI (Azure Container Instance).
For querying the endpoint, we can either use the REST call by importing the requests or by using the service run method with payload.
Here are the steps with REST call:
1. Store the scoring uri and primary key
2. Create the header with key "Content-Type" and value "application/json" and set the Authorization with Bearer token
3. Create the sample input and post to the requests.
Here is the sample input:
```
data= { "data":
       [
           {
               'Pregnancies': 6,
               'Glucose': 148,
               'BloodPressure': 72,
               'SkinThickness': 35,
               'Insulin': 0,
               'BMI': 33.6,
               'DiabetesPedigreeFunction': 0.627,
               'Age': 50
               
           },
           {
               'Pregnancies': 1,
               'Glucose': 85,
               'BloodPressure': 66,
               'SkinThickness': 29,
               'Insulin': 0,
               'BMI': 26.6,
               'DiabetesPedigreeFunction': 0.351,
               'Age': 31,  
           }
       ]
    }
```



## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response
Here is the screencast that demonstrated all the above mentioned process. Plese click on 
[link](https://www.youtube.com/watch?v=wGTl6yhKCxo&feature=youtu.be)

## Further Improvement
We can further improve the model by collecting more data and cleaning data so that the datas is balanced and not biased to any one. Also we can run AutoML for longer duration to try out different models. Also we can try different techniques for Hyperparameter tuning.
