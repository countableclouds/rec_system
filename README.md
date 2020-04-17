# Recommendation Algorithm
This algorithm is a deep-learning approach to collaborative filtering. Essentially, it involves taking the matrix of all rating interactions between user and problem. In this case, the only rating interaction is an upvote or no vote, which are 1 and 0 respectively. This matrix will usually be sparse, since most users don't rate most problems. 

However, if you can use this matrix to infer what the values actually are in the cells with zero, essentially filling in unknown spots, you can predict how much a user would like a problem.


## Neural Network Architecture
Our network architecture was modeled off of [this](https://www.ijcai.org/Proceedings/2017/0447.pdf) novel paper, which seemed to beat many of the benchmarks for collaborative filtering algorithms and therefore was optimal for our product.

This deep-learning approach used two neural network that each vectorize a user vector and a problem vector. The user vector consists of all the ratings the user has given to problems, and the problem vector consists of all the ratings it got from different users. Each network outputs the vectorization, and you can take the cosine similarity of those two vectors to get a "predicted rating" from 0 to 1. This is the normalized version of the total rating, and in this case, it actually is the "rating" since the maximum rating is 1. Although ratings are usually discrete and binary, this continuous version is more convenient for ranking problems for a user. These two networks are backpropagated together with a custom loss function that was determined by the paper to be optimal for training.

The interesting component is how to train in such a way that minimizes the amount of false information that is given by the lack of the rating. For this conundrum, in each epoch, the algorithm doesn't train on every single cell of the matrix, but rather on every cell with a positive rating, and some ratio of the no rating cells. This ratio is another hyperparameter of the net to be optimized, but well-chosen, it represents how much it matters if a user doesn't upvote a problem. With this consideration in mind, you obtain a randomized algorithm which in fact corrects for overfitting in part, and is a lot more accurate.

Note that this algorithm inherently takes into account a variety of factors, such as problem difficulty (some users may only rate easier challenges, and since many users like them will have done the same thing, most difficult problems will be zeroed out), genres of problems, and more. However, some explicit factors, such as tags, are used additionally to enhance the system further.


## Reconciling the Algorithm with a Problem-Ranking System
Now, the goal is to take this history and combine it with other benchmarks for how much a user likes a problem. Since there isn't one cohesive solution to do this, we came up with some of our own ideas to this end. Firstly, we decided that since we are using a tagging mechanism for our problems, we could find out how often a user said that they liked both of two tags. In doing so, we find a similarity measure between different tags, and therefore not only can we use the tags a user said they like, but also tags similar. Then, you can create a scoring scheme for a problem, rating how much a user would like it by taking into account how much the user likes each of the tags on the problem. This scoring scheme would only be evaluated on the top N problems that the deep-learning algorithm selected—the problems the user would rate the highest—and would be used to sort those.

Finally, because the user also subscribes to certain curators, those curators' problems would be invariantly placed at the top of the list of problems. Because users expect that their recommendations change very often, but they would only change once the entire system is retrained in this framework (as is necessary for collaborative filtering), there would also be a randomization aspect to the system, where noise is added to each predicted rating the ML algorithm produces. This noise changes the predicted ratings slightly, vastly changing the sorting but still delivering problems that the user likes. This system, although not yet implemented, is much more straightforward than the above, and would be implemented promptly in the future.

## Curators You Might Like

The curators you might like system, although also not implemented, follows from graph-search principles and the deep-learning algorithm. Firstly, since for each user we have its rating for each problem, curators that are well liked by users can be added to their feed. The other method is that since users follow certain curators, finding other followers of those curators and their corresponding subscriptions can allow for another good mechanism for identifying good people to follow.

Overall, these algorithms were interesting to devise, and hopefully they would helpfully allow users to find problems, of appropriate difficulty, that they'd truly enjoy.
