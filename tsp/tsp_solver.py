import sys
import math
import numpy as np
import random

evals = 0
budget = 0
dist = None
coords = []

class Solution:
    def __init__(self, permutation):
        self.permutation = permutation
        self.fitness = sys.float_info.max

def read_data(filename):
    global dist, coord
    lines = open(filename).readlines()
    #coords = []
    for line in lines:
        if line[0].isdigit():
            no, x, y = line.strip().split(" ")
            coords.append((float(x), float(y)))
    num = len(coords)
    dist = np.zeros(num ** 2)
    dist.shape = (num, num)
    for i in range(num):
        for j in range(num):
            dist[i][j] = math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
    return num

def improve(sol):
    global coords
    permu = np.copy(sol.permutation)
    for i in range(len(permu) -3):
        for j in range(i + 2, len(permu)-1):
            coeff = np.array([[coords[permu[i+1]][0] - coords[permu[i]][0],
                               coords[permu[j]][0] - coords[permu[j+1]][0]],
                              [coords[permu[i+1]][1] - coords[permu[i]][1],
                               coords[permu[j]][1] - coords[permu[j+1]][1]]])
            c = np.array([coords[permu[j]][0] - coords[permu[i]][0],
                coords[permu[j]][1] - coords[permu[i]][0]])
            try:
                x ,y = np.linalg.solve(coeff, c).tolist()
            except:
                continue
            if (x >=0 and x <=1) and (y >=0 and y <=1):
                newsol = two_opt_swap(sol, i+1, j)
                #print x, y
                if newsol.fitness < sol.fitness:
                    sol = newsol
                    #print "swapped"
    return sol

def two_opt_swap(sol, a, b):
    swap_perm = np.copy(sol.permutation)
    for i in range(a, b+1):
        swap_perm[i] = sol.permutation[b+a-i]
    swapped = Solution(np.asarray(swap_perm))
    evaluate(swapped)
    return swapped

def two_opt(sol):
    global evals, budget
    T = exp_cooling(10, 0.99999)
    num = len(sol.permutation)#read_data(filename)

    sol = Solution(np.random.permutation(range(num)))
    print sol.permutation
    evaluate(sol)
    best = sol

    while evals < budget:
        for i in range(0, num-1):
            for j in range(i+1, num):
                new = two_opt_swap(sol, i, j)
                t = T.next()
                p = P(best.fitness, new.fitness, t)
                if random.random() <= p:
                    best = new
        sol = best
        print sol.fitness

    return sol

def evaluate(sol):
    global evals
    evals += 1
    sol.fitness = 0
    for i in range(len(sol.permutation) - 1):
        sol.fitness += dist[sol.permutation[i]][sol.permutation[i+1]]
    sol.fitness += dist[sol.permutation[0]][sol.permutation[-1]]

def gen_neighbours(sol):
    neighbours = []
    i = 0
    while (i < len(sol.permutation) - 1 & evals < budget):
        new_order = np.copy(sol.permutation)
        temp = new_order[i]
        new_order[i] = new_order[i + 1]
        new_order[i + 1] = temp

        new_neighbour = Solution(new_order)
        evaluate(new_neighbour)
        neighbours.append(new_neighbour)
        i += 1
    return neighbours

def hc(filename):
    global evals, budget
    num, dist = read_data(filename)
    best = None
    while(evals < budget):
        current = Solution(np.random.permutation(range(num)))
        evaluate(current)
        while(evals < budget):
            neighbours = gen_neighbours(current)
            moved = False
            for neighbour in neighbours:
                if neighbour.fitness < current.fitness:
                    current = neighbour
                    print current.fitness
                    moved = True
                    break
            if not moved:
                break
        if best == None or best.fitness > current.fitness:
            best = current
    return best

def P(prev_score,next_score,temperature):
    if next_score < prev_score:
        return 1.0
    else:
        return math.exp(-abs(next_score-prev_score)*1./temperature)

def exp_cooling(start_temp,alpha):
    T=start_temp
    while True:
        yield T
        T=alpha*T

#def simulated_annealing(filename):
def usage():
    print "Usage: tsp_solver.py [filename] [budget]"
    print "Please retry"

if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
        exit()

    num = read_data(sys.argv[1])
    budget = int(sys.argv[2])
    sol = Solution(np.random.permutation(range(num)))

    sol1 = two_opt(sol)
    for i in sol.permutation[:-1]:
        print int(i),",",
    print int(sol.permutation[-1])
    print sol1.fitness
    sol1 = improve(sol1)
    for i in sol1.permutation[:-1]:
        print int(i), ",",
    print int(sol1.permutation[-1])
    print sol1.fitness
