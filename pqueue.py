import heapq as hq


class PQueue:
    def __init__(self):
        self.q = []

    def push(self, key, value):
        hq.heappush(self.q, (key, value))

    def pop(self):
        return hq.heappop(self.q)

    def repos(self, value, new_key):
        for i in range(len(self.q)):
            k, v = self.q[i]
            if v == value:
                self.q[i] = (new_key, value)
                hq.heapify(self.q)
                break
        else:
            assert False, "no such elem in priority queue"

    def __repr__(self):
        return "PQueue"

    def __str__(self):
        return str(self.q)

    def __bool__(self):
        return bool(self.q)
