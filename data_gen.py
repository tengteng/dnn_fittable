"""
Run
    python data_gen.py
then
    cut -d " " -f1 train.txt| sort -g | uniq -c
to see sample data label distribution
"""
import numpy as np
from math import sin, cos, gcd, exp, log2
from random import randrange
from sklearn.metrics import ndcg_score


def complex_mod(vec, tf=False):
    # not fittable
    score = (
        int(vec[0] * (vec[1] + vec[2]) * 17) % 7
        + int((vec[3] - vec[4]) * 11) % 3
        + int(sin(vec[5]) * cos(vec[6]) * 29)
        % (1 + int(sum([abs(x) for x in vec[7:15]])))
    ) % 3
    if not tf:
        score -= 1
    return score


def simple_mod(vec, tf=False):
    # not fittable
    score = int(abs(vec[0]) * 111) % int(abs(vec[1]) * 7 + 1)
    if score == 0:
        return score
    elif score in [1, 2]:
        return 1
    return 2


def simple_mod_2(vec, tf=False):
    # not fittable
    score = int(sum(vec) * 111) % 9
    return 0 if score < 3 else (1 if score < 6 else 2)


def simple_linear(vec, tf=False):
    # fc fittable
    score = (
        vec[0] * 100
        + vec[1] * 100
        - vec[2] * 10
        + vec[3] * 10
        + vec[4] * 20
        - vec[5] * 20
    )
    return 0 if score < -50 else (2 if score > 50 else 1)


def simple_linear_2(vec, tf=False):
    # fc fittable
    score = int(
        vec[0] * vec[1] / (abs(vec[2]) + 1) * vec[3] * 10 + vec[4] * 20 - vec[5] * 20
    )
    return 0 if score < -11 else (2 if score > 11 else 1)


def surf(vec, tf=False):
    # fc fittable
    Z = (
        1
        / 5
        * np.abs(1 - (vec[0] + vec[1] * 1j) ** -5)
        / np.abs(1 - 1 / (vec[0] + vec[1] * 1j))
    )
    Z = 20 * np.log10(Z)
    return 0 if Z <= -15 else (2 if Z >= 5 else 1)


def ifelse_logic(vec, tf=False):
    # fc fittable
    if vec[0] > vec[1]:
        return 0 if vec[2] > vec[3] else 1
    else:
        return 0 if vec[4] > vec[5] else 1


def has_gcd(vec, tf=False):
    # not fittable
    x = int(vec[0] * 100)
    y = int(vec[1] * 100)
    return 0 if gcd(x, y) == 1 else 1


def random_range(vec, tf=False):
    # fc fittable
    # Label Distribution: 5237 0 | 4763 1
    # FC: loss: 0.1644 - accuracy: 0.9287
    greater = int(max(vec[0], vec[1]) * 10) + 2
    lesser = int(min(vec[0], vec[1]) * 10)
    acc = 0
    for i in range(5):
        acc += randrange(lesser, greater)
    acc = acc // 5
    return 1 if acc >= 1 else 0


def bitcount(vec, tf=False):
    # not fittable
    # Label Distribution: 5514 0 | 4486 1
    # FC: loss: 0.6925 - accuracy: 0.5418
    # LSTM: loss: 0.7970 - accuracy: 0.5226
    x = int(vec[0] * 10)
    y = int(vec[1] * 10)
    diff = abs(bin(x).count("1") - bin(y).count("1"))
    return 1 if diff == 1 else 0


def ndcg(vec, tf=False):
    # fc fittable
    # Label Distribution: 4054 0 | 5946 1
    # FC: loss: 0.3006 - accuracy: 0.8628
    true_relevance = np.asarray([vec[:5]])
    scores = np.asarray([vec[5:10]])
    r = ndcg_score(true_relevance, scores)
    return 0 if r <= 0.5 else 1


def complex_func(vec, tf=False):
    # fc fittable
    # Label Distribution:  4193 0 | 5807 1
    # LSTM: loss: 0.3836 - accuracy: 0.8991
    s = (
        cos(exp(vec[0] * 10) - exp(vec[1] * 5)) ** 2
        + sin(vec[5] ** 2) ** 2
        + vec[6]
        - abs(vec[7] + vec[12])
        + (vec[8] * vec[9]) ** 2
    )
    s += log2(abs(vec[11]) + 0.1) ** 2 / 5
    return 1 if s < 1 else 0


FUNCS = {
    "complex_mod": complex_mod,
    "simple_mod": simple_mod,
    "simple_mod_2": simple_mod_2,
    "simple_linear": simple_linear,
    "simple_linear_2": simple_linear_2,
    "surf": surf,
    "ifelse_logic": ifelse_logic,
    "has_gcd": has_gcd,
    "random_range": random_range,
    "bitcount": bitcount,
    "ndcg": ndcg,
    "complex_func": complex_func,
}

if __name__ == "__main__":
    func = FUNCS["random_range"]
    with open("train.txt", "w") as fp:
        data = np.random.randn(10000, 20)
        labels = np.vstack(map(func, data))
        for l, f in zip(labels, data):
            fs = " ".join([f"{idx+1}:{ff}" for idx, ff in enumerate(f)])
            fp.write(f"{'+1' if l[0]==1 else l[0]} {fs}\n")

    with open("test.txt", "w") as fptest:
        data = np.random.randn(50000, 20)
        labels = np.vstack(map(func, data))
        for l, f in zip(labels, data):
            fs = " ".join([f"{idx+1}:{ff}" for idx, ff in enumerate(f)])
            fptest.write(f"{'+1' if l[0] == 1 else l[0]} {fs}\n")
