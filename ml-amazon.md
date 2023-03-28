## AWS Machine Learning
Machine learning, abbreviated as ML, has been a popular buzzword in the field of science and technology. 

### What is Machine Learning?
It is the art of teaching machines to learn without explicitly programming every rule for it to learn. We all would have been a part of teaching the machine as well as using the technology. 
We use Google Maps and whenever we don’t find the right address, we click on the back button indicating that the route which the Maps showed was wrong. This way, the Maps application learns to distinguish a wrong path and a right one. 
We use Google Assistant to book appointments, perform search and other operations, which uses Machine Learning to perform the action suggested by the user correctly. To be specific, it is a careful combination of Natural Language Processing and Machine Learning Algorithms. 

### Steps in implementing a machine learning algorithm:
When a user has to implement a machine learning algorithm, the below steps are usually taken:
1. Collect data and transform it. 
2. Pre-process the data.
3. Split the data into training and validation set.
4. Fit a relevant model on the data.
5. Train the data and get results. 
6. When the desired accuracy has been achieved, stop training, else repeat step 5. 

Amazon Machine Learning is a cloud based service, which helps developers build, train predictive models and use them in applications. These applications can be scaled and hosted on the cloud. It provides visualization tools which help build ML models, avoiding the need to write custom code as well as managing infrastructure. 

### Steps in Amazon Machine learning to build a model:
1. Prepare data
Datasets are available which are cleaned and in the right format. If you wish to use customized dataset, upload it to an Amazon S3 bucket. 

2. Creation of a training data source
Data source refers to an object which holds the location of input data and metadata about it. A data source can be created by providing access to the Amazon S3 location or providing schema details or providing name of attribute which needs to be predicted. The data source doesn’t hold the actual data, but just a reference to it. 
  - Open the ML console.
  - Click on ‘Get Started’. 
  - On the ‘Get started with Amazon Machine learning’ page, click on ‘Launch’.
  - Make sure ‘S3’ is selected on the ‘Input Data’->’Where is your data located’ option. 
  - In the S3 location, type in the full path of the input data location. 
  - Provide a value to the ‘Datasource name’.
  - Click on ‘Verify’. 
  - When the ‘S3 permissions’ dialog box appears, click ‘yes’. 
  - A page similar to the below snip can be seen if the data uploaded to S3 location can be accessed by Amazon ML. 
  - Establish a schema, by providing data types for all the column names. Amazon ML infers the schema, which needs to be checked (and changed if necessary). 

3. Create an ML model
  - The ‘Get Started’ wizard creates a training data source and a model. This takes the user directly to ‘ML model settings’. 
  - Make sure the model name is the same which you entered or the data source name.
  - In the ‘Training and evaluation settings’, make sure the ‘Default’ is selected.
  - Accept the default for ‘Name this evaluation’.
  - Click on ‘Review’, review the settings, and click on ‘Finish’. 
  - On the ‘ML model summary’ page, in the ‘ML model report’, click on ‘Evaluations->Evaluation->ML model->datasource_name->Summary’. 
  - In the ‘Evaluation summary’ page, review the evaluation data. 
  - Set a threshold by clicking on ‘Evaluation Summary -> Adjust score threshold -> Save threshold’.

4. Evaluation of the model
  - In the ‘ML model report’, click on ‘Try real-time predictions’. 
  - Paste a row of the data uploaded by clicking on ‘Paste a record -> Submit’. 
  - Click on ‘Create Prediction’. The prediction results can be seen on the right pane. 

## Conclusion
In this post, you understood how machine learning can be used to predict targets using Amazon Web Services. 





