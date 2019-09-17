import heapq

if __name__ == '__main__':
    arr = [2.11, 1.111, 1.112, 1.111, 1.122]
    print(list(arr))
    heapq.heapify(arr)
    print(list(arr))
    print(heapq.nsmallest(3, arr))
    a = heapq.heappop(arr)
    a = heapq.heappop(arr)
    b = heapq.heappop(arr)
    heapq.heappush(arr, a)
    print(list(arr))
    
