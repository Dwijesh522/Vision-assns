import heapq

class Tuple:
    def __init__(self, l, r):
        self.l = l
        self.r = r

class MyHeap(object):
    def __init__(self, initial=None, key=lambda x:x):
        self.key = key
        if initial:
            self._data = [(key(item), item) for item in initial]
            heapq.heapify(self._data)
        else:
            self._data = []
    def push(self, item):
        heapq.heappush(self._data, (self.key(item), item))
    def pop(self):
        return heapq.heappop((self._data)[1])
    def Nsmallest(self, x):
        return heapq.nsmallest(x, self._data)
def yo(t):
    return t.l

if __name__ == '__main__':
    heap = MyHeap(key=yo)
    one = Tuple(1, "one")
    two = Tuple(2, "two")
    three = Tuple(3, "three")
    four = Tuple(4, "four")
    heap.push(four)
    heap.push(three)
    heap.push(one)
    heap.push(two)
    print(heap.Nsmallest(3))
