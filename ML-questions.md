1. **Machine Learning Explainability using Permutation Importance**

Often, Machine Learning models are not fully understood by developers due to which the model becomes a black box. Developers generally understand it as a piece of software that takes data and applies some algorithm on it, generates patterns and gives out patterns/predictions on new data. Most of the time, developers are unaware of how the decisions are made, and what drives the predictions. It is difficult to gain insights from a model in general, despite its innumerable uses. 

An obvious query regarding a machine learning mode is determining the specific features in the dataset that have the most influence on the predictions. This factor is known as “feature importance”. This factor is evaluated using the “permutation importance” metric. 

Permutation importance can be computed after the model has been trained on the dataset. This factor checks about what happens on the metrics such as accuracy, precision, if all the data points of a single attribute are shuffled randomly in the test dataset or validation dataset, leaving the rest of the data as is. 

Hypothetically, reordering a column would result in reduced accuracy since the newly ordered data would have little to no correlation with the real-world metrics. 

Due to this, the accuracy of the model takes a dig. Since this is an important feature, it suffers the most when prominent features affecting the predictions are shuffled. 

2. **Introduction to Beam Search Algorithm**

Before jumping into the Beam search algorithm, let us understand the heuristic technique since the Beam search algorithm is a heuristic technique. Heuristic technique is a set of rules to determine which option would be most effective in achieving a specific goal. This technique helps improve the efficiency of a search process. 

Beam search is a heuristic search algorithm that checks a graph by extending the most promising node in a limited set of nodes. 

This technique expands on a specific number of best nodes at every level. It iterates over levels in a uniform fashion and moves downwards after going through every level’s nodes. 

It uses breadth-first-search to build the search tree. It generates the successors at the current level’s state for every level in the tree. But only a certain number of states are evaluated at every level. The remaining nodes are ignored.

The heuristic cost associated with every node is used to choose the best node at every level. The width of the beam search is denoted by a variable ‘W’, ‘B’ denotes the branching factor. At every depth, there will be ‘W x B’ nodes to consider, out of which only ‘W’ nodes will be considered. The width of the beam is reduced when a higher number of states are trimmed. 

When ‘W’ = 1, the search is a hill-climbing search wherein the best node is always chosen from the successor nodes. No states are disregarded if the beam width is unlimited. In such cases, the beam search becomes a breadth-first search. 

The beam width determines the amount of memory required to complete the search. This is at the cost of completeness and optimal nature of the graph solution. It is quite possible that the desired state was pruned/disregarded before it was reached.

**3. Install Tensorflow on Linux**

There are 4 commands to successfully install Tensorflow on Linux. You will understand how and why these need to be executed.

Software requirements:

\> Python 3.7 and <Python 3.10

Pip version 19.0 or greater

[Microsoft Visual C++ Redistributable for Visual Studio 2015,2017 and 2017](https://support.microsoft.com/help/2977003/the-latest-supported-visual-c-downloads)

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) for GPU support

To install Miniconda:


|curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86\_64.sh -o Miniconda3-latest-Linux-x86\_64.sh|
| :- |

When prompted, enter ‘Yes’.

Execute:

|bash Miniconda3-latest-Linux-x86\_64.sh|
| :- |

Restart your terminal/source ~/.bashrc. This will enable the conda command.

To test if Miniconda was installed successfully, execute:


|conda -V|
| :- |

Create a conda environment:

The ‘–name’ indicates the name of the conda environment.

|conda create --name tf python=3.9|
| :- |

To activate the environment:

|conda activate tf|
| :- |

To deactivate the environment:

|conda deactivate|
| :- |

The below section is required if you have GPU requirements:

Install [NVIDIA GPU Driver](https://www.nvidia.com/Download/index.aspx).

To verify the NVIDIA GPU Driver installation,

|nvidia-smi|
| :- |

To install CUDA and cuDNN with conda:

|conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0|
| :- |

You can configure the system path every time you start a new terminal once you activate your conda environment.

|export LD\_LIBRARY\_PATH=$LD\_LIBRARY\_PATH:$CONDA\_PREFIX/lib/|
| :- |

You can automate the process of configuring your system path:


|<p>mkdir -p $CONDA\_PREFIX/etc/conda/activate.d</p><p>echo 'export LD\_LIBRARY\_PATH=$LD\_LIBRARY\_PATH:$CONDA\_PREFIX/lib/' > $CONDA\_PREFIX/etc/conda/activate.d/env\_vars.sh</p>|
| :- |


You will need recent pip version so upgrade pip:

|pip install --upgrade pip|
| :- |

You can Install Tensorflow with pip:

|pip install tensorflow|
| :- |

You can verify Tensorflow installation:

|python3 -c "import tensorflow as tf; print(tf.reduce\_sum(tf.random.normal([1000, 1000])))"|
| :- |

Successful installation returns a tensor.

You can verify GPU setup:

|python3 -c "import tensorflow as tf; print(tf.config.list\_physical\_devices('GPU'))"|
| :- |

Successful set up returns a list of GPU devices.

**4. Difference between TensorFlow and Theano**

In this article, you will understand the significant differences between Tensorflow and Theano libraries. 

[Tensorflow](https://tensorflow.org/)

**Pros**:

- It is an open-source end-to-end platform to build machine learning applications. 
- It was developed by researchers and developers at Google Brain.
- It is a general framework, and can be applied to other domains too.
- It provides Python, and C++ APIs.
- It has comprehensive and flexible tools, libraries, and community to build and deploy state-of-the-art Machine Learning applications.
- It is available on Linux, Windows, Android, iOS, and macOS.
- It also provides support for reinforcement learning, deep learning, NLP, image recognition, time series, and video detection.
- It has excellent documentation, and a supportive community for contributors. - It provides parallelism in terms of data and models. 
- It supports execution on CPU and GPU.

**Cons:**
- Matrix operations can’t be performed.
- It takes time to execute operations in comparison to other frameworks.
- Dynamic typing is prone to errors in high scalability development. 

[Theano](http://deeplearning.net/software/theano/)

**Pros:**
- It is a Python library that allows you to define, optimise, and evaluate mathematical expressions.
- These expressions involve multi-dimensional arrays, and Theano works on them efficiently. 
- It has an efficient integration with NumPy, hence numpy.ndarray works well with Theano-compiled functions. 
- It allows GPU usage, and helps perform data intensive computations.
- It also evaluates derivatives with one or multiple inputs.  
- It evaluates expressions quickly since it generates C code dynamically. 
- Irrespective of the value used in the mathematical expression, it evaluates and provides precise solutions. 

**Cons:**
- Debugging is tough since error messages are huge. 
- Large compilation time for complex models, which makes it difficult to maintain and work on them. 

**5. Difference between TensorFlow and Keras**

In this article, you will understand the significant differences between Tensorflow and Keras libraries. 

[Tensorflow](https://tensorflow.org/)

**Pros**:
- It is an open-source end-to-end platform to build machine learning applications. 
- It was developed by researchers and developers at Google Brain.
- It is a general framework, and can be applied to other domains too.
- It provides Python, and C++ APIs.
- It has comprehensive and flexible tools, libraries, and community to build and deploy state-of-the-art Machine Learning applications.
- It is available on Linux, Windows, Android, iOS, and macOS.
- It also provides support for reinforcement learning, deep learning, NLP, image recognition, time series, and video detection.
- It has excellent documentation, and a supportive community for contributors. - It provides parallelism in terms of data and models. 
- It supports execution on CPU and GPU.

**Cons:**
- Matrix operations can’t be performed.
- It takes time to execute operations in comparison to other frameworks.
- Dynamic typing is prone to errors in high scalability development. 

**Keras**
- It is an open-source neural network library that has the ability to run on Theano and Tensorflow. 
- It helps construct deep learning algorithms and other Machine Learning algorithms.
- It has an API with user-friendly features that can be easily understood. 
- You can choose Keras to pick any library that it supports for the backend.
- It provides pre-trained models that help users to improve models further.
- It has a strong community of users that help contribute and improve on models.

**Cons:**
- Some of the pre-trained models don’t provide a lot of support to design models. 
- Errors given out are not easily understandable. 
- It is a low-level API.

**6. Save and load models in Tensorflow**

You can save models in tensorflow when the model is being trained or even after the training has been completed. You can avoid training models for long periods of time, by splitting up the training time, and Tensorflow ensures to resume the model from where it left off. 

Saving models helps understand how the model works, track your progress, and can be used by other user’s as their baseline model as well with new data. 

You can save the model in different ways, and we will look at one of them in this article.

After you create a model, evaluate it, load the weights, and re-evaluate the model, you can save it.

You can call the [tf.keras.Model.save](https://www.tensorflow.org/api_docs/python/tf/keras/Model#save) to save the model's architecture, weights, and training configuration in a single file/folder. This helps export a model so it can be used without access to the code. Since the optimizer-state is recovered, you will have the ability to resume training from where you left off.

A model can be saved in two different file formats: **SavedModel** and **HDF5**. 

The TensorFlow SavedModel format is the default file format in Tensorflow version 2.x. Saving a fully-functional model can further be loaded into TensorFlow.js and trained and run in web browsers.

The SavedModel format serialises models. Such saved models can be restored using [tf.keras.models.load_model](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model). They are compatible with TensorFlow Serving too. This format is a directory that contains protobuf binary and a Tensorflow checkpoint. You can load the model:

|new\_model = tf.keras.models.load\_model('saved\_model/my\_model')|
| :- |
You can save the model using below commands:

|<p>!mkdir -p saved\_model</p><p>model.save('saved\_model/my\_model')</p>|
| :- |

**7. Impact and Example of Artificial Intelligence**

Artificial intelligence deals with programming computers to detect patterns in new data, make decisions based on user input, and give output based on the user input. All the rules are not explicitly defined and it is expected by the developer that the machine learns these rules by experience, using a reward-punishment mechanism. 

Artificial intelligence has come a long way with self-driving cars, bots, object and facial recognition, and much more. 

In recent times, algorithms have been developed with a better accuracy that can help provide outputs which have better accuracies. 

Due to the improvements in the field of Artificial intelligence, the number of trivial human interactions (customers communicating with service agents) have reduced due to chatbots, improved and better quality healthcare, object recognition and more. 

Improved user experience: Instead of talking to agents about trivial attributes, the human resources can be used efficiently. These have been replaced by chatbots. 

Home assistants: Applications like Alexa are built which use voice based inputs to produce outputs.

Home automation: Using Internet of Things, and Machine Learning, based on the user’s movement, objects inside the home can be controlled.

Object recognition: In automated traffic detection, objects can be identified with better accuracy in real-time. 

Self-driving cars: Companies like Tesla are milking with the developments in Machine Learning by automating car driving.

Robots: Robots can be used for dangerous tasks instead of humans.

Reduction in errors: Since computers are better with calculation, the amount of errors has reduced.

**8. Logs and Metrics**

In this article, we will understand more about logs and metrics.

Both logs and metrics are characteristics associated with data. 

**Metrics:**

It helps measure the functionality of the component, and defines thresholds required for usage. It helps assess service value, and provide a continuous view of the entire environment. Multiple metrics can be used to evaluate the quality of an application, hence it is important to choose metrics wisely. 

Metrics such as throughput and response time are usually used with most applications, whereas much specific metrics like clicks per second or clicks per advertisement are for higher-end use cases. 

Metrics are crucial to understand the application better. They are easy to implement, but need to be scaled carefully when the infrastructure grows. 

To obtain data for every metric, an event needs to be generated for every occurrence of the event in question. Designing and implementing such events for every development cycle is a tedious task. This includes the overhead of memory and service uptime. 

Metrics need to be carefully created since using a lot of metrics is an overkill.

In all, they help identify trends, understanding system deficiencies, and performance.

**Logs**

Metrics alone is not sufficient to understand applications better. Logs shine emphasis on specific events and errors associated with them. They help record everything for monitoring purposes, and preserve most of the information regarding events. 

Such log data helps investigate incidents when they go wrong, as well as help perform root cause analysis. 

Logs are unique for every application, and they can be structured based on the insights required to be extracted. Regular expressions can be used to parse these log data or a unified logging layer can be implemented due to which log data can be viewed as JSON data. Logs can be analysed and used to identify security breach attempts, misuse of application’s functions, and maintenance of records for legal compliance requirements. 

They can’t be easily stored and need to be implemented carefully so as to not lead to loops and infinite conditions. Relevant error messages need to be mentioned so that errors can actually be resolved by looking at logs of the error in the application. 

**9. Difference between TensorFlow and Caffe**

[Tensorflow](https://tensorflow.org/)

- It is an open-source end-to-end platform to build machine learning applications. 
- It was developed by researchers and developers at Google Brain.
- It is a general framework, and can be applied to other domains too.
- It provides Python, and C++ APIs.
- It has comprehensive and flexible tools, libraries, and community to build and deploy state-of-the-art Machine Learning applications.
- It is available on Linux, Windows, Android, iOS, and macOS.
- It also provides support for reinforcement learning, deep learning, NLP, image recognition, time series, and video detection.
- It has excellent documentation, and a supportive community for contributors. - It provides parallelism in terms of data and models. 
- It supports execution on CPU and GPU.
- Matrix operations can’t be performed.
- It takes time to execute operations in comparison to other frameworks.
- Dynamic typing is prone to errors in high scalability development. 

**Caffe**
- It is an open source framework developed by keeping expression, speed, and modularity in mind.
- It is developed by community contributors and Berkeley AI Research. 
- It can be used with CPU and GPU.
- It is used in research and industrial development applications.
- It doesn’t have a steep learning curve.
- It works well with images in deep learning. 
- It is easy to dive into and models can be explored easily. 
- It aims to help developers who want to have first hand experience with deep learning. 
- It can process more than 60M images in a day.
- It can be used with Linux, Windows and MacOS.

**10. Introduction to TensorFlow Lite**

It is a mobile library designed to deploy models on mobile, microcontrollers and edge devices. It comes with tools that enable on-device machine learning on mobile devices using 5 aspects: latency, privacy, connectivity, size, and power consumption. 

- It provides support on Android, iOS, embedded Linux and microcontrollers.
- It supports multiple languages such as Java, Swift, Objective C, C++, and Python. 
- It also provides hardware acceleration and model optimization. 
- The documentation provides end-to-end examples for machine learning projects such as image classification, object detection, question answering, pose estimation, text classification, and many more on different platforms. 
- There are two aspects to developing a model in Tensorflow Lite:
    - Building a Tensorflow Lite model;
    - Running inference. 

A Tensorflow Lite model is represented in a portable format known as FlatBuffers, i.e a .tflite file extension. It has reduced size and quick inference that enables TensorflowLite to execute efficiently on divides that have limited compute and memory resources. 

It also includes metadata about model, pre- and post processing pipelines which is in human-readable format.

Inference is the process of executing a Tensorflow Lite model on-device which helps make predictions on new data. It can be done in two ways depending on if the model has metadata or not. 

With metadata: You can use out-of-the-box API or build custom inference pipelines. Using Android devices, you can generate code wrappers using Android Studio ML Model Binding or Tensorflow Lite Code Generator. 

Without metadata: You can use the Tensorflow Lite Interpreter API which is supported on multiple platforms.

**11. Knowledge based agents in AI**

Knowledge-based agents represent searchable knowledge that can be reasoned. These agents maintain an internal state of knowledge, take decisions regarding it, update the data, and perform actions on this data based on the decision. Basically, they are intelligent and respond to stimuli similar to how humans react to different situations.

Examples: Based on the user's question (that behaves as the external stimuli), they provide an answer from their knowledge base (the data warehouse where they store basic knowledge) that provides a satisfactory answer to the user’s question. 

It has the below-mentioned features:

- Knowledge base (KB)
    - It is one of key components of a knowledge-based agent. It stores facts and data pertaining to the real-world.

- Inference Engine(IE)
    - It is a knowledge-based system engine that helps infer new knowledge from the existing data within the system.

- Actions performed by an agent
    - When the knowledge-based agent needs to be updated, the inference system comes into picture. It uses a ‘Ask-Tell’ mechanism wherein new data is inferred from pre-existing data. Agent has a knowledge base which contains base knowledge that performs certains actions when it is called. 

- Actions performed by the knowledge base Agent
    - It ‘Tells’ its recognitions from its environment and imparts to the knowledge base what it requires.

- It ‘Asks’ the knowledge base what actions to perform. It receives answers from the knowledge base. Based on the action selected, the agent executes the action.

A knowledge base can use two approaches:

Declarative Approach: Starting with an empty knowledge base, the agents ‘tells’ or fills the knowledge base with data.

Procedural Approach: The necessary behaviours are directly converted into code in an empty knowledge-base.


**12. Why TensorFlow is So Popular ñ Tensorflow Features**

Tensorflow is an open source Machine learning framework that helps develop models, train pre-trained models by providing high level APIs. It was developed by researchers and developers at Google Brain. It is a general framework, and can be applied to other domains too. The pre-trained models can be used quickly for experimenting and production environments. It can be used by beginners in the field of ML to researchers and academic projects. 

**Features of Tensorflow**

- It provides Python, and C++ APIs.
- It has comprehensive and flexible tools, libraries, and community to build and deploy state-of-the-art Machine Learning applications.
- It is available on Linux, Windows, Android, iOS, and macOS.
- It also provides support for reinforcement learning, deep learning, NLP, image recognition, time series, and video detection.
- It has excellent documentation, and a supportive community for contributors. 
- It provides parallelism in terms of data and models. 
- It supports execution on CPU and GPU.
- Tensorflow provides Tensorflow Lite which is a mobile library designed to deploy models on mobile, microcontrollers and edge devices. It comes with tools that enable on-device machine learning on mobile devices using 5 aspects: latency, privacy, connectivity, size, and power consumption. 

**Features of Tensorflow Lite**

- It supports multiple languages such as Java, Swift, Objective C, C++, and Python. 
- It also provides hardware acceleration and model optimization. 
- The documentation provides end-to-end examples for machine learning projects such as image classification, object detection, question answering, pose estimation, text classification, and many more on different platforms.

**13. Difference between YOLO and SSD**

YOLO and SSD are real-time object detection systems that possess significant differences, that have been listed below:

**YOLO (You Only Look Once)**

- It uses a neural network to help with real-time object detection. 
- It became popular due to its speed and accuracy. 
- It is considered a regression problem, where the algorithm looks at the object/s only once. There are algorithms associated with YOLO that achieve 155 FPS (frames per second). Image is divided into a grid, and every grid calculates class probabilities and bounding box parameters to determine the object in its entirety. 
- It is an open-source detection technique that works with images and videos. 
- It is preferred when the object size is small as well.
- It can be used with self-driving cars, and other salient applications of artificial intelligence.


**SSD (Single Shot Detector)**
- It works well with real-time object detection. 
- It discretizes the output space of the bounding boxes into a couple of default boxes.
- These default boxes are of different ratios and scales per feature map location. 
- The network generates a score for the presence of every object category in every default box and produces adjustments to the box. 
- These adjusts are made to match the shape of the object.
- The network combines the predictions from different feature maps with different resolutions thereby handling objects of varying sizes gracefully. 
- The speed is a result of eliminating the bounding box proposals and feature resampling.
- This includes a convolutional filter that predicts object categories and offsets in the bounding box locations using filters (separate predictors) for different sized objects. 

**14. What is No Free Lunch Theorem**

The ‘No-Free Lunch’ theorem (NFL) is usually used in the field of optimization and machine learning. The theorem states that all optimizations perform equally well when the performance of every algorithm is calculated as an average of all possible problems. 

This means there is no single ‘the-best’ optimization algorithm. This implies that there is no single best machine learning algorithm to use in predictive modelling problems.

This doesn’t mean all algorithms are equal. In practice, all algorithms are NOT created equal. This is because the entire machine learning related problems comes as a theoretical concept under the NFL theorem. This set of problems is larger than the actual practical machine learning problems that practitioners attempt to solve. 

Some algorithms perform better than others on certain types of data, and every algorithm comes with its own pros and cons. These pros and cons are a result of the assumptions associated with the algorithm. 

For example: Neural networks perform well with complicated tasks such as object detection and image classification, but it is possible that it may suffer from overfitting if not trained with the right weights and biases. 

Ensure that you understand the ins and outs of the machine learning algorithm you are trying to use, the data available and how well the algorithm performs with such kind of data. 

Models are as good as the assumptions made with them, and the data associated with it while training. 

Simple models such as logistic regression have bias and end up underfitting, whereas complex models such as neural networks have variance and end up overfitting.

A right model for your problem statement would be somewhere in between two bias-variance extremes. 

The best way to find a model for your data is to experiment with data models, and see which gives the best/relevant result. The end result of every model can be compared to each other using a cross-validation strategy.

**15. Top 10 Benefits of Machine Learning in Businesses**

The field of machine learning is developing rapidly, and the benefits are innumerable. Some of the benefits in the field of business have been listed below.

**Improved consumer experience:** Human efforts to interact with consumers have reduced and bots are used instead. This has helped with the automated and trivial responses that are required at times. Human resources can be used for better tasks as well.

**Resource planning:** Since patterns can be detected from data, the resources required can be planned in advance, and can be utilised efficiently. This way, machine learning makes businesses cost effective too.

**Effective predictions:** Predictions based on sales in businesses can be more accurate leading to better outcomes. It can also help manage resources during difficult times.

**Adapting to market changes:** Since predictions can yield results based on historical data, it can help adapt to the changes in the market, which could be due to a variety of reasons. 

**Improved data security:** Machine learning can help detect fraud before it happens, and prevent fraudulent transactions thereby saving businesses from heavy losses. There threats can be of a wide variety depending on the nature of the business.

**Data management:** An important aspect of business is to manage the ever-flowing data in a better way, use it well and extract insights from it so that it can be used for the benefit of the organisation. Unsupervised machine learning algorithms do just that. They don’t need to be explicitly told to extract specific patterns, since they work well on unstructured data. 

**Better decision making:** When data is put to good use and meaningful insights are extracted from it, it can help take better decisions that would eventually improve the revenue. 

**Pattern recognition for better profits:** Patterns in data can be recognized that would help improve the revenue of the business. 

**Eliminate manual efforts:** Trivial tasks can be automated hence human resources can be used for smarter tasks, thereby reducing the manual efforts on things that require less to no attention. 

**Spam detection:** Similar to data security, machine learning algorithms can help differentiate between useful and useless data (in the form of emails, and more). This ensures that businesses don’t waste resources on filtering data, rather than putting the relevant data to good use.

**Medical diagnosis:** Healthcare organisations can use sophisticated tools to yield accurate results in a shorter span of time depending on the medical history of the person.


**16. Dilated and Global Sliding Window Attention**

Models such as BERT and SpanBERT can be used to perform NLP based, with the limitation that they use self-attention. These transformer based models don’t perform well when long texts are involved. To overcome this limitation, a long-document transformer was introduced. 

Longformer is a modified transformer architecture that consists of a self-attention component that can detect long texts. This is still not very efficient hence a number of attention models are introduced to improve the accuracy of the model. Sliding window attention model can be used to improve the accuracy, wherein two variations will be discussed in this case. 

**Sliding Window Attention**

It is an attention based model which parses a ‘m’ x ‘n’ resolution image that has a fixed step size. This fixed step size is used to capture the target image efficiently thereby improving the efficiency of the longformer.

There are two types of sliding window attention models:

**Dilated sliding window attention**: A dilation on top of sliding window improves the coverage of the input image, and keeps the computational costs intact. It helps parse small texts where the dilation rate is low, and can also parse long texts with a higher dilation rate.

**Global sliding window attention**: It deals with task-specific situations where a specific part of text has to be detected from the text input. It is symmetric in nature and helps find specific sequences by taking into consideration all the tokens along with the row/column present in the input.

**17. Scaling techniques in Machine Learning**

The technique of scaling generates infinite values on the objects to be measured. These techniques help understand the relationship between the objects. 

Rank order: One item is judged against the rest of the objects.

The respondents contain several objects who rank/order the objects based on a criteria.

Rank order scaling is ordinal in nature, that is, (n-1) scaling decisions are made in this technique. 

**Constant sum scaling:**

In this technique, a constant sum of units is assigned to the respondent. For example, if a specific number of points indicate the importance of the product. 

If the attribute is unimportant, the respondent assigns 0 to it.

If an attribute is twice as important as another attribute, it receives twice as many points.

The sum of all points is constant, that is 100, hence the scale name.

**Non-comparative scales:**

In non-comparative scales, every object of the data set is scaled independently. The resulting data is assumed to be ratio scaled.

Continuous rating scales: 

1. It is a graphic continuum.
1. It generally has two coordinated extremes.
1. Easy to construct. 
1. Simple to use.
1. The respondent rates the object by placing a mark on a continuous line.
1. The extreme values aren’t predefined.

Itemised rating scales: 

1. It is a graphic continuum.
1. It has two coordinated extremes.
1. It is easy to use.
1. It is easy to construct. 
1. The respondent rates the object based on a number or brief description associated with every category.
1. The categories are ordered on scale position. 
1. Hence, the respondents pick the specific category that describes the object in question.

**18. What ís Text Annotation and its Types in Machine Learning?**

Text annotation identifies and labels sentences with metadata to define characteristics of sentences. This could be highlighting parts of speech, grammar, phrases, keywords, emotions, and so on depending on the project. The better the quality and quantity of data, the better the model performs. 

In this article, you will understand different text annotation methods.

1\. Sentiment Annotation

Based on the emotion/sentiment associated with the response, the text is annotated. Sarcasm filled text should be understood as is, rather than being termed negative or positive. Sentiment is an important aspect here. Every sentence needs to be labelled based on the available options. 

2\. Intent Annotation

This tells about the intent of the user, i.e when interacting with bots, users respond with different intentions. Some want to complain, discuss, ask for redemption, and so on. The different types of desires have to be captured accurately by the models. 

3\. Entity Annotation

This is the most important text annotation technique, which is used to identify, tag, and attribute multiple entities in a given text or sentence. We could break down entity annotation further into the following:

4\. Text Classification

It is also known as document classification or text categorization. In this method, annotators read paragraphs or sentences and understand the sentiments, emotions, and intentions behind them. These textual phrases are classified based on the comprehension specified by the respective project. 

5\. Linguistic Annotation

Linguistic annotation is a hybrid of the above discussed features, but this is done in a specific language. It involves phonetics annotation where intonations, natural pauses, stress, and other parts of the speech associated with the language are tagged too.  

**19. Fuzzy Logic and Probability : The Confusing Terms**

In this article, you will understand the difference between fuzzy logic and probability.

Fuzzy logic

1. It is a many-valued logic where the truth value of variables may be a real number between 0 and 1, including 0 and 1. 
1. Everything is associated with a degree.
1. It is based on natural language processing. 
1. It can be integrated with programming. 
1. It is best suited for approximation. 
1. It is generally used by quantitative analysts.
1. It helps understand the concept of vagueness.
1. It captures the meaning of partial truth.
1. The degree of membership is in a set.
1. It is used in air conditioners, facial pattern recognition, vacuum cleaners, transmission systems, and subway control systems. 

Probability

1. It is a branch of mathematics that deals with numerical descriptions of how likely an event is supposed to occur, or how likely a proposition is true.”
1. The value lies in the range of 0 and 1.
1. It can’t be used for high approximations. 
1. It captures partial knowledge.
1. It deals with the likelihood of the occurrence of an event.                  
1. It can’t capture any type of uncertainty.
1. It can only give a numerical value of whether an event can occur, and how many chances that it would be possible. 
1. The probability event is in a set.
1. It is used in the field of manufacturing, decision-making, risk evaluation, scenario analysis, long-term gains and losses calculation.

Both are terms to express uncertainty but in different ways. It informs you how uncertainty changes over a period of time when new data comes in. 

Consider the following 2 examples:

Example 1: There is a 59 percent chance that there will be a storm today.

Example 2: The weather will be stormy today.

Example 1 suggests a percentage/probability that is a number. Example 2 is fuzzy, there is no clarity about the severity. The statement is subjective.

**20. How to Setup Anaconda For Data Science?**

[Anaconda](https://www.anaconda.com/) is a distribution package for Python and R. It can be used to install Python in an isolated environment. It is easy to install and has more than 1000 data science packages for download. 

You can install Python on Linux, Windows, and MacOS. 

After installing Anaconda on your platform, you will see a navigator, the Anaconda navigator. 

It is a graphical UI that is automatically installed with Anaconda. It has multiple features like Jupyter Notebook, RStudio, and so on. 

You can install packages using Anaconda with a simple command like:


|conda install package\_name|
| :- |

1\. Jupyter Notebook

Jupyter Notebook is a web-based, interactive competing notebook environment. It helps you edit and run docs. It is an open-source web application that helps you create and share documents with live code, equations, and visualisations. 

2\. JupyterLab

It provides an environment for interactive and reproducible computing. It is based on the Jupyter Notebook.It helps work with documents, text editors, terminals, and custom components in a flexible manner.

3\. Spyder

It is a Python IDE which is open-source and cross-platform. It is known as the scientific development IDE. It is lightweight, and integrates with packages such as Matplotlib, SciPy, NumPy, Pandas and so on. 

4\. RStudio

It is an integrated development IDE for R. It provides programming tools that help work with R scripts, outputs, text, and images.

**21. Consensus clustering**

Clustering is an unsupervised learning algorithm that doesn't have a labelled response variable to train the algorithm on. Hence it is im[ortant to find similarities between observations based on the dataset. 

Consensus clustering is also known as aggregated clustering which is a robust approach that relies on multiple iterations of a chosen clustering algorithm based on subsamples of the dataset. Inducing sampling variability using sub-sampling provides metrics to assess the stability of the clusters and parameter decisions. 

As the name suggests, consensus clustering gains a consensus on the observation’s cluster assignment. 

Why use consensus clustering?

K means clustering algorithm uses random initialization procedure that results in different cluster results in every iteration of the algorithm. The value of K needs to be initialised. Hence clustering depends on many metrics and may produce biased clusters that could be unstable. 

Steps to implement consensus clustering:

It is important to decide on the number of iterations, N.

Choose a set of K values to test.

For every K, iterate N number of times, and create a set of clusters for the observations. 

Since we segment the data, not all observations are clustered in every iteration. 

To get a consensus matrix, look at the pairwise relation between every user. It is a dissimilarity matrix of size N x N. 

Advantages of consensus clustering:

- High-quality clusters
- Produces right number of clusters
- Handles missing data well.
- Individual partitions can be independently obtained. 

**22. Latent semantic analysis**

Latent semantic analysis is a natural language processing method used in the field of search engine optimization for information retrieval. It analyzes the relationship between a set of documents and the terms present within them. 

It uses singular value decomposition which is a mathematical technique. It helps scan unstructured data and finds relationships between concepts and terms. 

It uses searching and automated document categorization.

It can be used for text summarization, search engine optimization, and other applications.

Disadvantages

It can’t capture multiple meanings of a word. 

The vector representation gives an average of the word’s meaning in the corpus. 

This makes it difficult to compare documents. 

Text data suffers from high dimensionality which can be reduced using latent semantic analysis (LSA). It reformulates text data in terms of hidden features.
