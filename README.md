# Earth

This project is one mini Youtube recommendation system.

Based on TensorFlow, it implements the famous paper <Deep Neural Networks for YouTube Recommendations>.

It uses dbpedia data,treats the documents as users, and the document's classification as the video.

This means we know the features(information) of the users,so we use them to infer the video which the users will play.

The candidates generation progress implement the multi classification based on softMax.

The ranking  progress implement the 0-1 classification based on logistic regression.
