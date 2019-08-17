class stochastic_node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.parent = None
class stochastic_list:
    def __init__(self, head_val, size, forward_threshold):
        self.head = None
        self.current = None
        self.forward_threshold = forward_threshold
        self.size = size
        # initializing list of stochastic nodes
        self.node_list = []
        temp = stochastic_node(head_val)
        self.head = temp
        self.current = self.head
        self.node_list.append(temp)

        for i in range(1, size):
            temp = stochastic_node(-1)
            self.current.next = temp
            temp.parent = self.head
            self.node_list.append(temp)
            self.current = self.current.next

        self.current.next = self.head
        self.current = self.current.next
    def check(self, node_val):
        change = abs(node_val - self.head.data)
        if change >= self.forward_threshold:
            if(self.current.next == self.head):
                self.head.data = self.current.data/(self.size-1)
                self.current = self.head
            else:
                if(self.current == self.head):
                    self.current = self.current.next
                    self.current.data = node_val
                else:
                    self.current.next.data = self.current.data + node_val
                    self.current = self.current.next
        else:
            self.current = self.head
    def learned_val(self):
        return self.head.data
if __name__ == '__main__':
    sl = stochastic_list(12, 4, 1)
    print(sl.learned_val())
    sl.check(14)
    print(sl.learned_val())
    sl.check(14)
    print(sl.learned_val())
    sl.check(13)
    print(sl.learned_val())
    sl.check(14)
    print(sl.learned_val())
