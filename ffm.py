import numpy as np
from pqueue import PQueue
from enum import Enum
import math


class State(Enum):
    FAR = 1
    NB = 2  # narrow band
    FROZEN = 3


def _check(p: (int, int), mesh):
    return 0 <= p[0] < len(mesh) and 0 <= p[1] < len(mesh[0])


def _neighbours(p: (int, int), mesh):
    ps = [(p[0] - 1, p[1]),
          (p[0] + 1, p[1]),
          (p[0], p[1] - 1),
          (p[0], p[1] + 1)]

    return list(filter(lambda p: _check(p, mesh), ps))


def _val(color):
    *cs, _ = color
    cs = list(map(lambda c: 1 - c if c < 0.5 else c, cs))
    return sum(cs) / 3


def _f(x: (int, int), mesh):
    return _val(mesh[x])


def _distance(p: (int, int), mesh, t, state):
    def choose(p1: (int, int), p2: (int, int), *_, default=float('inf')):
        if _check(p1, mesh) and state[p1] == State.FROZEN and _check(p2, mesh) and state[p2] == State.FROZEN:
            return (t[p1], p1) if t[p1] <= t[p2] else (t[p2], p2)
        elif _check(p1, mesh) and state[p1] == State.FROZEN:
            return t[p1], p1
        elif _check(p2, mesh) and state[p2] == State.FROZEN:
            return t[p2], p2
        return default, None

    up = p[0] - 1, p[1]
    down = p[0] + 1, p[1]
    ta, pa = choose(up, down)

    left = p[0], p[1] - 1
    right = p[0], p[1] + 1
    tb, pb = choose(left, right)

    tm, pm = (ta, pa) if ta <= tb else (tb, pb)
    if pm is None:
        return tm

    fprev = _f(pm, mesh)
    fcurr = _f(p, mesh)
    dif = fcurr + 1 / (2 * abs(fcurr - fprev) + 1) - 1  # TODO
    assert abs(dif) > 1e-5, dif

    def default():
        return min(ta, tb) + 1 / dif

    if pa is None:
        assert pb is not None
        return default()
    elif pb is None:
        return default()

    d = (ta + tb) ** 2 - 2 * (ta ** 2 + tb ** 2 - 1 / dif ** 2)
    if d > 1e-10:
        return (ta + tb + math.sqrt(d)) / 2
    return default()


def fmm(mesh, start, *_, fillcolor=None, field=None):
    fillcolor = [0, 1, 0, 1] if fillcolor is None else fillcolor
    field = min(mesh.shape[:2]) / 10 if field is None else field

    t = np.array([[float('inf')] * mesh.shape[1] for _ in range(mesh.shape[0])])
    state = np.array([[State.FAR] * mesh.shape[1] for _ in range(mesh.shape[0])])
    pq = PQueue()

    # init
    t[start] = 0
    state[start] = State.FROZEN
    for vn in _neighbours(start, mesh):
        d = _distance(vn, mesh, t, state)
        t[vn] = d
        state[vn] = State.NB
        pq.push(d, vn)

    # compute time
    while pq:
        vdist, v = pq.pop()
        t[v] = vdist
        state[v] = State.FROZEN
        for vn in _neighbours(v, mesh):
            d = _distance(vn, mesh, t, state)
            if state[vn] == State.FAR:
                state[vn] = State.NB
                pq.push(d, vn)
            elif state[vn] == State.NB:
                pq.repos(vn, d)

    # results
    for row in range(t.shape[0]):
        for col in range(t.shape[1]):
            if t[row][col] <= field:
                assert state[row][col] == State.FROZEN
                mesh[row][col] = fillcolor
