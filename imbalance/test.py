import numpy as np

def funk(tup):
    if tup[0] == -1:
        return 0
    else:
        return 1

    # print(tup[0], tup[1])

def main():
    N = 100
    l = []
    for i in range(N):
        for j in range(N):
            if i > j:
                l.append((i, j))
            
            else:
                l.append((-1, -1))

    a = list(map(funk, l))

    print(np.array(a).reshape(N, N))

if __name__ == '__main__':
    main()