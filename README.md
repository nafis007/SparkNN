Microblogging websites are a common platform where people all around the world can
share their opinions and views on various issues and topics. Here we proposed to analyze
geolocation of Twitter user at the time of uploading the post based on the content of the
message. Correctly predicting geolocation of a tweet is useful for a wide range of applications.
Since a considerable number of tweets are generated every second, we need a robust system
that can process these tweets in real time. Here using batch processing has its limitations;
therefore we are using Apache Storm, a real-time and fault-tolerant system. In the last few
years, the rise of popularity of machine learning especially deep learning is incredible in both
academic and industry. However, writing a deep learning algorithm from scratch is probably
beyond the skill set of most people want to explore the field. It is much more efficient to
utilize the tremendous resources available in different deep learning toolkit. Therefore we
used Deeplearing4j with our streaming platform to build a distributed neural network. We
carefully looked at the issues regarding high commutations of building a neural network
and use Apache Spark for task distribution. As the data is available gradually over time,
we are following an incremental machine learning approach to and whether a tweet is from
Melbourne or not. Our model found an accuracy of 91.43% of correctly predicting the tweets
from Melbourne while testing in real time.

Task Distribution:

Mohammad Nafis Ul Islam
Task Contribution: This member mainly focused on DeepLearning4J framework and used
it to implement the Multilayer Neural Network and the Distributed Learning Method, using
Spark Platform on it. Besides merging the learner code with the Storm code, he also analyzed
different characteristics of the initial raw dataset and extracted specific tweets, which were later
used for feature extraction. He also did the analysis of the learning model, by running different
experiments, using the pre-processed data; and finally came up with the results after choosing
an optimal learner, based on the analysis.
