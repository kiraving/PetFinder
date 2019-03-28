import pandas as pd
import numpy as np

# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

p442 = pd.read_csv('train_442_preds.csv')
p456 = pd.read_csv('train_456_preds.csv')
p459 = pd.read_csv('train_459_preds.csv')

print(quadratic_weighted_kappa(p442.target_442.values, p442['442'].values))
print(quadratic_weighted_kappa(p456.target_456.values, p456['456'].values))
print(quadratic_weighted_kappa(p459.target_459.values, p459['459'].values))

w1 = 0.25
w2 = 0.25
w3 = 0.45
mean = (p442['442'].values * w1 + p456['456'].values * w2 + p459['459'].values * w3) / (w1 + w2 + w3)
mean = np.round(mean).astype(int)
print(quadratic_weighted_kappa(p442.target_442.values, mean))


def to_optimize(w1, w2, w3):
    mean = (p442['442'].values * w1 + p456['456'].values * w2 + p459['459'].values * w3) / (w1 + w2 + w3)
    mean = np.round(mean).astype(int)
    return quadratic_weighted_kappa(p442.target_442.values, mean)


from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import os


# Bounded region of parameter space
pbounds = {
    'w1': (0.0, 1.0),
    'w2': (0.0, 1.0),
    'w3': (0.0, 1.0)
}

optimizer = BayesianOptimization(
    f=to_optimize,
    pbounds=pbounds,
    random_state=1,
)

if os.path.isfile("./logs_weight_selection.json"):
    optimizer = load_logs(optimizer, logs=["./logs_weight_selection.json"])

logger = JSONLogger(path="./logs_weight_selection.json")
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

optimizer.maximize(
    init_points=10,
    n_iter=1500,
)
