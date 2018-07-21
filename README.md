<h1 align="center">
  <img src="media/rb-logo.png" width="20%"><br/>RetailBox
</h1>

<h4 align="center">
  🛍️ Machine Learning eCommerce Recommender System
</h4>

## Overview

RetailBox is a recommender system that uses machine learning to make recommendations based off of a user's past purchase history. This project was inspired by Amazon and their recommender system techology. In this system we use Collaborative Filtering based off of implicit user data.

## Dataset

The dataset we used is the [UCI Online Retail Dataset](http://archive.ics.uci.edu/ml/datasets/online+retail). This dataset has 25900 transactions, with each transaction containing 20 features and these transactions were made by 4,300 users.

## Training/Test Split

The way we split the data was by time. In this scenario, our system makes recommendations based on implicit user data from the past so this is how we split our data.

## Context of Recommendation Scenario

In this recommender system, the context we used can be seen by the diagram below. Our main model to generate recommendations for a user and if a user bought an item our system recommended, then it was a success!

<p align="center">
  <img src="https://i.imgur.com/5WegTbB.png">
</p>

## Installation

```
pip install retailbox
```

## CLI

```
$ retailbox --help

  🛍️ Machine Learning eCommerce Recommender System

  Usage
    $ retailbox [<options> ...]

  Options
    --help, -h                Display help message
    --search, -s              Search customer by ID [Can be any integer 0-4338]
    --customer, -c <int>      Input customer ID [Can be any integer 0-4338]
    --info, -i                Display customer information
    --status, -s              Display process info
    --list, -l                List available customer IDs
    --version, -v             Display installed version

  Examples
    $ retailbox --help
    $ retailbox --search
    $ retailbox --customer 1028
    $ retailbox -c 1028
    $ retailbox -m 1028 --info
    $ retailbox -m 1028 -i --status
```

## Usage

```
from retailbox.recommender import recommender

customer = 2874  # Customer ID
status = True   # Display status information of program while running

# Generate recommendations
recommender(
    customer_id=customer,
    status=status)

```

## References

* [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
* [Matrix Factorization For Recommender Systems](https://joyceho.github.io/cs584_s16/slides/mf-16.pdf)
* [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618.pdf)
* [Based Subreddit Recommender System](https://cole-maclean.github.io/blog/RNN-Based-Subreddit-Recommender-System/)
* [A Recurrent Neural Network Based Recommendation System](https://cs224d.stanford.edu/reports/LiuSingh.pdf)
* [BPR in Python](https://github.com/gamboviol/bpr)
* [SGD and ALS in collaborative filtering - CrossValidated Thread](https://stats.stackexchange.com/questions/201279/comparison-of-sgd-and-als-in-collaborative-filtering)
* [Python Pickle Module for saving Objects by serialization](https://pythonprogramming.net/python-pickle-module-save-objects-serialization/)
* [Python Packaging from Init to Deploy](https://www.youtube.com/watch?v=4fzAMdLKC5k)

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
