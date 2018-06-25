<h1 align="center">
  <img src="media/rb-logo.png" width="20%"><br/>RetailBox
</h1>

<h4 align="center">
  🛍️ Machine Learning eCommerce Recommender System
</h4>

## Overview

RetailBox is a recommender system that uses machine learning to make recommendations based off of a user's past purchase history. This project was inspired by Amazon and their recommender system techology.

## Dataset

The dataset we used is the [UCI Online Retail Dataset](http://archive.ics.uci.edu/ml/datasets/online+retail). This dataset has 25900 transactions, with each transaction containing 20 features, and these transactions were made by 4,300 users.

Here are the features of the dataset:

* Invoice No
* Stock Code
* Description
* Quantity
* UnitPrice
* CustomerID
* Country

## Training/Test Split

The way we split the data was by time.

## Algorithms Used

* SGD Matrix Factorization
* Recurrent Neural Networks

## Performance Measures

We set a baseline measurement for our recommender system, and then compared it to a recommender system that used matrix factorization and RNNs.
Baseline

SGD Matrix Factorization

RNN

The RNN performed the best.

## Challenge and Workarounds

* Dataset had several rows of a single transactions
* We don't know what the user doesn't like.
* Matrix Factorization does not take

## Context of Recommendation Scenario

In this recommender system, the context we used can be seen by the diagram below. Our main model to generate recommendations for a user and if a user bought an item our system recommended, then it was a success!

<p align="center">
  <img src="https://i.imgur.com/5WegTbB.png" width=500>
</p>

## Installation

## CLI


## Usage


## References

* [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
* [Matrix Factorization For Recommender Systems](https://joyceho.github.io/cs584_s16/slides/mf-16.pdf)
* [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618.pdf)
* [Based Subreddit Recommender System](https://cole-maclean.github.io/blog/RNN-Based-Subreddit-Recommender-System/)
* [A Recurrent Neural Network Based Recommendation System](https://cs224d.stanford.edu/reports/LiuSingh.pdf)


## Team

Coded by:

* [Mohsin Baig](https://github.com/moebg)
* [Shivank Mishra](https://github.com/shivankmishra)

## Attribution and Inspiration

* Icon by [Korawan Markeaw](https://thenounproject.com/korawan_m/)
* Inspired by [MovieBox](https://github.com/klauscfhq/moviebox) project by Mario Sinani & Klaus Sinani

## Contributing

Contributions are always welcome! For bug reports or requests please submit an issue.

## License

[MIT](https://github.com/moebg/retailbox/blob/master/LICENSE)