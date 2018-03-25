# Multimedia
Implementation of a Diverse Image Retrieval system capable of retrieving relevant images from Flickr in a way that increases the diversification of the results. This implementation surpasses Flickr search engine and the best system submitted in MediaEval 2017 in Diverse Image Retrieval task. 

# Requirements
To run the code you have to install this packages:

- keras (https://keras.io/)
- theano (http://deeplearning.net/software/theano/)
- RankNet (https://github.com/shiba24/learning2rank)
- numpy (http://www.numpy.org/)
- pickle (https://docs.python.org/2/library/pickle.html)


# Run
To run the code you need to run the `main.py` file in root directory:

```
python main.py
```

After reading the data, at the end, performance results including `P@20`, `CR@20` and `F1@20` are listed for four different solutions:

- Ranking the results withoun learning
- Ranking the results with Attention Mechanism
- Ranking the results with RankNet
- Ranking the results with Attention and RankNet


