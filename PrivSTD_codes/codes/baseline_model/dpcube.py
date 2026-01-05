from ugag import UG
import numpy
import math


def bucketcost(X, l1, l2):
    # this function used to compute the intermediate costs of buckets

    p = []
    pp = []
    for i in range(l1 + 1):
        if i == 0:
            p = p + [numpy.zeros(l2 + 1)]
            pp = pp + [numpy.zeros(l2 + 1)]
            continue

        x1 = numpy.zeros(l2 + 1)
        x2 = numpy.zeros(l2 + 1)
        for j in range(l2):
            x1[j + 1] = x1[j] + X[i - 1][j]
            x2[j + 1] = x2[j] + X[i - 1][j] ** 2

        p = p + [x1]
        pp = pp + [x2]

    for i in range(l1):
        for j in range(l2 + 1):
            p[i + 1][j] = p[i][j] + p[i + 1][j]
            pp[i + 1][j] = pp[i][j] + pp[i + 1][j]

    return p, pp


def Compute(p, pp, x0, y0, x1, y1):
    # this function used to compute the cost of bucket (x0,y0,x1,y1)
    a1 = pp[x1 + 1][y1 + 1] + pp[x0][y0] - pp[x0][y1 + 1] - pp[x1 + 1][y0]
    a2 = p[x1 + 1][y1 + 1] + p[x0][y0] - p[x0][y1 + 1] - p[x1 + 1][y0]

    return a1 - a2 ** 2 * 1.0 / ((x1 - x0 + 1) * (y1 - y0 + 1))


def dpcube(epsilon, p, pp, rp, X2, start, end, delta):
    # this function used to compute the noisy counts
    (x0, y0) = start
    (x1, y1) = end

    len0 = x1 - x0 + 1
    len1 = y1 - y0 + 1

    sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    if len0 > len1:
        bias = Compute(p, pp, x0, y0, x1, y1)
        cur = bias + util.old_div(1.0, epsilon)
        flag = False
        pos = x0

        for k in range(x0, x1):
            bias1 = Compute(p, pp, x0, y0, k, y1)
            bias2 = Compute(p, pp, k + 1, y0, x1, y1)

            if bias1 + bias2 + util.old_div(2.0, epsilon) < cur:
                cur = bias1 + bias2 + util.old_div(2.0, epsilon)
                flag = True
                pos = k

        if flag:
            dpcube(epsilon, p, pp, rp, X2, (x0, y0), (pos, y1), delta)
            dpcube(epsilon, p, pp, rp, X2, (pos + 1, y0), (x1, y1), delta)
        else:
            ncnt = rp[x1 + 1][y1 + 1] + rp[x0][y0] - rp[x0][y1 + 1] - rp[x1 + 1][y0] + numpy.random.normal(0.0, sigma)
            navg = ncnt * 1.0 / (len0 * len1)

            for i in range(x0, x1 + 1):
                X2[i][y0:y1 + 1] = navg

    else:
        bias = Compute(p, pp, x0, y0, x1, y1)
        cur = bias + util.old_div(1.0, epsilon)
        flag = False
        pos = y0

        for k in range(y0, y1):
            bias1 = Compute(p, pp, x0, y0, x1, k)
            bias2 = Compute(p, pp, x0, k + 1, x1, y1)

            if bias1 + bias2 + util.old_div(2.0, epsilon) < cur:
                cur = bias1 + bias2 + util.old_div(2.0, epsilon)
                flag = True
                pos = k

        if flag:
            dpcube(epsilon, p, pp, rp, X2, (x0, y0), (x1, pos), delta)
            dpcube(epsilon, p, pp, rp, X2, (x0, pos + 1), (x1, y1), delta)
        else:
            ncnt = rp[x1 + 1][y1 + 1] + rp[x0][y0] - rp[x0][y1 + 1] - rp[x1 + 1][y0] + numpy.random.normal(0.0, sigma)
            navg = ncnt * 1.0 / (len0 * len1)

            for i in range(x0, x1 + 1):
                X2[i][y0:y1 + 1] = navg

    return X2


def DPCube(x, epsilon, delta, alpha=0.5):
    l1, l2 = x.shape

    X1 = UG(x, eps=alpha * epsilon, sigma=math.sqrt(2 * math.log(1.25 / (alpha * delta))) / (alpha * epsilon))
    X2 = numpy.ndarray((l1, l2), 'float32')
    X2.fill(0)

    rp, rpp = bucketcost(x, l1, l2)

    p, pp = bucketcost(X1, l1, l2)

    X2 = dpcube((1 - alpha) * epsilon, p, pp, rp, X2, (0, 0), (l1 - 1, l2 - 1), (1 - alpha) * delta)

    return X2
