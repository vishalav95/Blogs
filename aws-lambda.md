## Amazon Web Services Lambda 
The AWS Lambda is a cloud service provided by Amazon, which is also known as ‘Lambda’ in short. It is a part of the Amazon Web Services, referred to as ‘serverless compute’. The word ‘serverless compute’ means the service doesn't explicitly require a server to run the code. It manages the servers without the interference from the developer. 
The word ‘serverless’ doesn’t mean there are no servers/ no code. The user can simply focus on writing high-quality, bug-free code. The requirements (such as providing infrastructure, resources, scaling, storage, server) are facilitated by Amazon Lambda service.
Serverless also means the server doesn’t need to run at all in time. For example- When there is no request, the server doesn’t need to run. This way, compute resources are not wasted. The server appears when required and disappears when its work is done. 
The developers can write their code and upload it to the Lambda service which takes care of all the requirements to run the code. The gruesome task of deciding the storage or selecting resources to run the code are eliminated. 

### Terminologies used with Lambda service
The code by developers is executed inside the Lambda service known as ‘Lambda function’. It is executed only when called. This call could be a trigger or a simple function call or an HTTP request or occurrence of an event (external or internal) or a new document uploaded to S3, a scheduled job or a simple notification. 

#### Salient features of AWS Lambda
- Allows developers to focus on writing efficient and high-quality code. 
- Manages AWS resources required to execute the code. 
- Initiates and terminates instances. 
- Scales the resources up and down (it is elastic- which means it can be easily scaled up).
- Provides software patches and updates automatically handled by the service. 
- Pay-as-you-go, i.e pay for the services consumed. 
- It is event-driven.
- It is stateless.
- It is serverless

Now let us see how each of these characteristics contributes to the behaviour of Lambda.
- Event driven
This means the emphasis is on executing the code efficiently by automatically providing the resources required.
- Serverless
The developer doesn’t have to provide or manage the servers, storage or resources required to run their code. 
It provides abstract server management, hence the developers focus on writing quality. As new tasks are introduced in the code, Lambda service works towards scaling (up or down) it.
 - Stateless
The output of the function inside the Lambda service depends only on the parameters passed to the method. 
- Language Support
The developers can write code in multiple language, such as Python, Java, Node.js, Go and C#. 
It supports compilers like Maven, Gradle and function building packages. 

### What kind of code is written and executed with Lambda?
Lambda service is primarily used as a trigger to execute other programs from AWS services periodically. 
For example: Suppose we need to fetch data from different sources regularly, the data can be loaded into a database by triggering a program/code written inside the Lambda service. This Lambda service can be programmed to trigger at certain times of the day. 
The code executed in Lambda can be a response to other web services in Amazon, such as creating an object in Amazon Store Service (S3) bucket. 

### How to code inside AWS Lambda:
- A handler is the point of entry inside a Lambda service. Usually, the data to a handler is in JSON format. The output is also in the JSON format. 
- The runtime environment needs to be specified. Example if the developer uses Python, a Python runtime environment is provided. 
- Write the code in a stateless manner, i.e the output depends on the parameters passed to the method executed inside the Lambda service. 
- All the method variables should be declared inside the handler. 
- Lambda service execute code based on a certain condition. For this, it should have the ‘+rx’ (read and execute) permissions. 
- Delete methods that are not used by the Lambda service. 

### The pricing of AWS Lambda
It is a pay-as-you-use service. This means the user pays for the resources they consume. The charges are based on the number of requests made to the Lambda function as well as the amount of time during which the code is executed. 
A request is counted when the function starts its code execution or it is triggered by other external/internal events. The first one million requests are not charged. Every other request is charged as $0.20 per request. 
The time for the code to get executed is the moment the code begins execution and ends when the function is terminated or completed. This pricing is based on the amount of memory allocated to the Lambda function. 
They have a free tier that can be used as a sample to understand how Lambda could be used in conjunction with AWS EC2 instance, CloudWatch and other Amazon Web Services. 
Consider the below example which is executed on Lambda:
- The input is in JSON format.
{
“num_1” : 40.0,
“num_2”: 100.0
}
- Suppose our end result needs to be the difference, remainder and sum of the inputs. Observe that the output is also in JSON format. 
{
“num_1” : 40.0,
“num_2”: 100.0,
“sum”: 140.0,
“difference”:60,
“remainder”:2.5
}
- Sign up for a free trial of the AWS by creating or signing in to your account. 
- We will be using Python as our coding language to execute methods within Lambda. Since Python is a scripting language, it can be coded within the AWS console. 
- Go to the ‘Lambda’-> ‘Functions’->’Create Functions’.
- Click on the ‘Author from scratch’ and enter all the basic information required (See image below).
- Enter the runtime. Here, we are using ‘Python 3.7’.
- Click on ‘Create Function’ to create a new function. 
- Select on ‘Select a test event’ and select ‘Configure test events’.
- Create a new test event, provide a name to the Event name and enter the inputs mentioned in step 1. Now click on ‘Create’. It can be observed that the Event Template has been filled in prior to us entering anything (‘Hello World’). Instead of this, we can enter the data which we specified in step 2 to see the difference, remainder and sum of the two inputs. 
- This event can be tested by clicking on the ‘Test button’. The output would be ‘Hello World’. 
- The default output (Hello World) can be seen in the below snip. Also, it is in the JSON format. 

Note: Third party logging APIs are supported by Lambda and can be achieved with the help of Amazon API Gateway Service.

### Design specifications of AWS Lambda
These can’t be called limitations, since they have been designed to be what they are. Such boundaries have been discussed below:
- Hardware limitations, such as disk size being limited to 512 MB.
- The execution time out can be 5 minutes maximum. 
- Data sent through a ‘GET’ or ‘PUT’ request can’t be more than 6MB. 
- The size of the type of request, headers, and other metadata can’t be more than 128 KB. 

## Conclusion
In this post, we saw the significance of AWS Lambda and how it can be used to execute code without the user worrying about the resources, storage and infrastructure. 


 
