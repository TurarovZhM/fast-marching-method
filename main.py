import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pqueue import PQueue
from enum import Enum


class State(Enum):
    FAR = 1
    NB = 2  # narrow band
    FROZEN = 3


def neighbours(p: (int, int), mesh):
    for x in range(p[0] - 1, p[0] + 2):
        if not 0 <= x < len(mesh):
            continue
        for y in range(p[1] - 1, p[1] + 2):
            if y < 0 or y >= len(mesh[0]):
                continue
            if (x, y) != p:
                yield (x, y)


def distance(p: (int, int), mesh):
    return 1


def fmm(mesh, start, *_, fillcolor=None, field=None):
    fillcolor = [0, 0, 1, 1] if fillcolor is None else fillcolor  # TODO: is needed?
    field = len(mesh) // 10 if field is None else field

    dist = np.array([[float('inf')] * len(mesh[0]) for _ in range(len(mesh))])
    status = np.array([[State.FAR] * len(mesh[0]) for _ in range(len(mesh))])
    pqueue = PQueue()

    # init
    dist[start] = 0
    status[start] = State.FROZEN
    for v in neighbours(start, mesh):
        dst = distance(v, mesh)
        if status[v] != State.NB:
            status[v] = State.NB
            pqueue.push(dst, v)
        else:
            # TODO assert that we are decreasing dst
            pqueue.repos(v, dst)

    # loop
    while pqueue:
        dst, v = pqueue.pop()
        status[v] = State.FROZEN
        dist[v] = dst

        for vn in neighbours(v, mesh):
            if status[vn] == State.FROZEN:
                continue
            d = distance(vn, mesh)
            if status[vn] != State.NB:
                status[vn] = State.NB
                pqueue.push(d, vn)
            else:
                pqueue.repos(vn, d)

    # result
    assert len(mesh) == len(dist)
    assert len(mesh[0]) == len(dist[0])  # TODO better way to get inner size?
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
    fmm(img, x)
    plt.imshow(img)

    plt.show()


if __name__ == "__main__":
    main()
