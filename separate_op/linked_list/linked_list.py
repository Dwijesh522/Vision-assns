class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

# first in first out
class Linked_list:          
    def __init__(self):
        self.first = None
        self.median = None
        self.last = None
        self.size = 0
        self.delta = 0
    def push_back(self, node):
        if self.first is  None:
            self.first = node
            self.median = self.first
            self.last = node
            self.size = 1
            self.delta = 0
        else:
            temp = self.last.data - node.data
            if(temp <0):
                temp *= -1
            self.delta += temp

            self.last.next = node
            self.last = self.last.next

            self.size += 1
            
            if(self.size % 2 == 1):
                self.median = self.median.next
    def top1(self):
        return self.first.data
    def top2(self):
        return self.first.next.data
    def get_size(self):
        return self.size
    def last1(self):
        return self.last.data
    def delete_front(self):
        if(self.size == 1):
            self.size = 0
            self.last = None
            self.first = None
            self.median = None
            self.delta = 0
        else:
            temp = self.first.data - self.first.next.data
            if(temp < 0):
                temp *= -1
            self.delta -= temp

            self.first = self.first.next
            if(self.size %2 == 0):
                self.median = self.median.next
            self.size -= 1
    def get_median(self):
        return self.median.data
    def get_delta(self):
        return self.delta
    def destruct_list(self):
        self.size = 0
        self.last = None
        self.first = None
        self.median = None

if __name__ == "__main__": 
    ll = Linked_list()
    one = Node(1)
    two = Node(3)
    three = Node(1)
    four = Node(3)
    five = Node(1)
    six = Node(3)
    ll.push_back(one)
    ll.push_back(two)
    ll.push_back(three)
    ll.push_back(four)
    ll.push_back(five)
    ll.push_back(six)
    ll.delete_front()
    print(ll.get_delta())

