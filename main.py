import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pqueue import PQueue
from enum import Enum
import math


class State(Enum):
    FAR = 1
    NB = 2  # narrow band
    FROZEN = 3


def check(p: (int, int), mesh):
    return 0 <= p[0] < len(mesh) and 0 <= p[1] < len(mesh[0])


def neighbours(p: (int, int), mesh):
    lst = [(p[0] - 1, p[1]),
           (p[0] + 1, p[1]),
           (p[0], p[1] - 1),
           (p[0], p[1] + 1)]

    return list(filter(lambda p: check(p, mesh), lst))


def val(color):
    r, g, b, _ = color
    return r + g + b  # TODO


def distance(p: (int, int), mesh, status):
    def value(x: (int, int)):
        return val(mesh[x])

    def choose(p1: (int, int), p2: (int, int), *_, default=0):
        if check(p1, mesh) and status[p1] == State.FROZEN and check(p2, mesh) and status[p2] == State.FROZEN:
            return min(value(p1), value(p2))
        elif check(p1, mesh) and status[p1] == State.FROZEN:
            return value(p1)
        elif check(p2, mesh) and status[p2] == State.FROZEN:
            return value(p2)
        return default

    up = p[0] - 1, p[1]
    down = p[0] + 1, p[1]
    va = choose(up, down)

    left = p[0], p[1] - 1
    right = p[0], p[1] + 1
    vb = choose(left, right)

    assert va or vb  # TODO

    d = (va + vb)**2 - 2 * (va**2 + vb**2 - 1)
    if d >= 0:
        return (va + vb + math.sqrt(d)) / 2
    return min(va, vb) + 1  # TODO
    # return float('inf')


def fmm(mesh, start, *_, fillcolor=None, field=1.0):
    fillcolor = [0, 0, 1, 1] if fillcolor is None else fillcolor  # TODO: is needed?

    dist = np.array([[float('inf')] * len(mesh[0]) for _ in range(len(mesh))])
    status = np.array([[State.FAR] * len(mesh[0]) for _ in range(len(mesh))])
    pqueue = PQueue()

    # init
    dist[start] = 0
    status[start] = State.FROZEN
    for v in neighbours(start, mesh):
        dst = distance(v, mesh, status)
        if status[v] != State.NB:
            status[v] = State.NB
            dist[v] = dst
            pqueue.push(dst, v)
        else:
            assert dst <= dist[v]
            pqueue.repos(v, dst)

    # loop
    while pqueue:
        dst, v = pqueue.pop()
        status[v] = State.FROZEN
        dist[v] = dst

        for vn in neighbours(v, mesh):
            if status[vn] == State.FROZEN:
                continue
            d = distance(vn, mesh, status)
            if status[vn] != State.NB:
                status[vn] = State.NB
                pqueue.push(d, vn)
            else:
                # assert dist[vn] >= d  # TODO
                pqueue.repos(vn, d)

    # result
    assert len(mesh) == len(dist)
    assert len(mesh[0]) == len(dist[0])  # TODO: better way to get inner size?
    for row in range(len(dist)):
        for col in range(len(dist[0])):
            if dist[row][col] <= field:
                mesh[row][col] = fillcolor


def main():
    img = mpimg.imread('img1.png')
    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    x = plt.ginput(1)[0]
    x = (int(round(x[1])), int(round(x[0])))
    fmm(img, x, field=0.9)
    plt.imshow(img)

    plt.show()


if __name__ == "__main__":
    main()
