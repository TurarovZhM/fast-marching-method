import heapq as hq


class PQueue:
    def __init__(self):
        self.q = []

    def push(self, key, value):
        hq.heappush(self.q, [key, value])

    def pop(self):
        return hq.heappop(self.q)

    def repos(self, value, new_key):
        for i in range(len(self.q)):
            if self.q[i][1] == value:
                break
        else:
            assert False, "No such key in priority queue"
        self.q[i][0] = new_key
        hq._siftdown(self.q, 0, i)  # accessing protected member is all right

    def __repr__(self):
        return "PQueue"

    def __str__(self):
        return str(self.q)

    def __bool__(self):
        return bool(self.q)
