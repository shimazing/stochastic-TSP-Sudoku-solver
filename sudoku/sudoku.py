#!/usr/bin/env python

import sys
import numpy as np
import copy
import random
from collections import Counter

evals = 0
budget = 0
fixed = {}
grid_remainder_list = []
pool = []


class Solution:
    def __init__(self, representation):
        self.representation = representation
        self.fitness = sys.maxint

#selection
class Tournament:
    def select(self, population, num):
        best = None
        best_idx = 0
        for _ in range(num):
            i = random.randrange(len(population))
            if (best == None) or population[i].fitness < best.fitness:
                best = population[i]
                best_idx = i
        return best

def compare(a1, a2):
    for i in range(9):
        for j in range(9):
            if a1[i][j] != a2[i][j]:
                return False
    return True

grid_starts = [(0, 0), (0, 3), (0, 6), (3, 0), (3, 3), (3, 6) ,(6, 0), (6, 3),(6, 6)]


# heuristic preprocessing
def preprocessing(rep):
    global pool
    for i in range(9):
        tmp = []
        for j in range(9):
            if rep[i][j] == 0: tmp.append(set(range(1,10)))
            else: tmp.append(set([rep[i][j]]))
        pool.append(tmp)

    for i in range(9):
        for j in range(9):
            if len(pool[i][j]) == 1:
                num = list(pool[i][j])[0]
                grid_idx = i - i%3 + j/3
                x, y = grid_starts[grid_idx]

                for k in range(9):
                    if len(pool[i][k]) > 1: pool[i][k] -= set([num]) # row
                    if len(pool[k][j]) > 1: pool[k][j] -= set([num]) # col

                for r in range(3):
                    for s in range(3):
                        if len(pool[x+r][y+s]) > 1: pool[x+r][y+s] -= set([num]) # grid
    while True:
        pool_ = copy.deepcopy(pool)
        for i in range(9):
            x, y = grid_starts[i]
            tmp = []
            for r in range(3):
                for s in range(3):
                    tmp += list(pool_[x+r][y+s])
            tmp = Counter(tmp)
            unique = []
            for j in tmp:
                if tmp[j] == 1:
                    unique.append(j)

            if not unique:
                break
            for uni in unique:
                idx = None
                for r in range(3):
                    for s in range(3):
                        if uni in pool_[x+r][y+s]:
                            idx = (x+r, y+s)
                            pool_[x+r][y+s] = set([uni])

                for k in range(9):
                    r, s = idx
                    if len(pool_[r][k]) > 1: pool_[r][k] -= set([uni]) # row
                    if len(pool_[k][s]) > 1: pool_[k][s] -= set([uni]) # col

                for r in range(3):
                    for s in range(3):
                        if len(pool_[x+r][y+s]) > 1: pool_[x+r][y+s] -= set([uni]) # grid

        if compare(pool, pool_):
            break
        pool = pool_

    for i in range(9):
        for j in range(9):
            if len(pool[i][j]) == 1: fixed[(i,j)] = list(pool[i][j])[0]

    print pool
    print fixed
    return pool, fixed


def mutate(indiv):
    rep = indiv.representation[:]
    rand_grid = random.randrange(0,9)
    x, y = grid_starts[rand_grid]
    not_fixed = []
    for r in range(3):
        for s in range(3):
            if (x+r, y+s) not in fixed:
                not_fixed.append((x+r, y+s))
    random.shuffle(not_fixed)
    a , b = not_fixed[0], not_fixed[1]
    rep[a[0]][a[1]] , rep[b[0]][b[1]] = rep[b[0]][b[1]], rep[a[0]][a[1]]
    new = Solution(rep)
    eval(new)
    return new


def cross_over(indiv1, indiv2):
    rand_idx = random.randrange(1, 9)

    rep1 = indiv1.representation[:]
    rep2 = indiv2.representation[:]

    for i in range(rand_idx):
        x, y = grid_starts[i]
        for r in range(3):
            for s in range(3):
                rep1[x+r][y+s] , rep2[x+r][y+s] = rep2[x+r][y+s], rep1[x+r][y+s]

    new1 = Solution(rep1)
    new2 = Solution(rep2)

    eval(new1)
    eval(new2)

    return new1, new2

def reshaping(indiv):
    reshaped = []
    for i in range(9):
        reshaped.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
    for i, grid in enumerate(indiv):
        x, y = grid_starts[i]
        for j in range(9):
            r, s = x + j/3, y + j%3
            reshaped[r][s] = grid[j]
    return reshaped

def init_individuals(line, p_num):
    global  grid_remainder_list
    print "Initializing ...."
    assert(len(line) == 81)

    line = [int(i) if i != "." else 0 for i in line]

    rep = np.array(line).reshape(9,9)
    pool, fixed = preprocessing(rep)
    print fixed
    print pool[2][0]
    starts = [0, 3, 6, 27, 30, 33, 54, 57, 60]
    grid_list = []

    for x, y in grid_starts:
        grid = []
        grid_remain = set(range(1,10))
        for i in range(3):
            for j in range(3):
                if len(pool[x+i][y+j]) == 1:
                    grid.append(list(pool[x+i][y+j])[0])
                    grid_remain -= pool[x+i][y+j]
                else:
                    grid.append(0)
        grid_list.append(grid)
        grid_remainder_list.append(list(grid_remain))
    #print grid_list

    population = []
    for i in range(p_num): #make p_num population
        indiv = random_gen()

        #print indiv.fitness
        population.append(indiv)
        population = sorted(population, key = lambda x: x.fitness)
    print "End Initialization!"
    return population



def eval(indiv):
    global evals
    evals += 1
    fitness = 0
    row_list = np.array(indiv.representation)
    col_list = np.transpose(row_list)
    for i in range(9):
        fitness += 9 - len(set(row_list[i]))
        fitness += 9 - len(set(col_list[i]))
    indiv.fitness = fitness

# generate one random solution
def random_gen():
    global pool

    rand = []

    for i in range(9):
        rand.append([0, 0, 0, 0, 0, 0, 0, 0, 0])

    for i in range(9):
        x, y = grid_starts[i]
        while True:
            permu = np.random.permutation(grid_remainder_list[i])
            k = 0
            for j in range(9):
                r, s = x + j/3, y + j%3
                if (r,s) not in fixed:
                    rand[r][s] = permu[k]
                    k += 1
                else: rand[r][s] = fixed[(r,s)]

            flag = True
            for j in range(9):
                r, s = x + j/3, y + j%3
                if rand[r][s] not in pool[r][s]:
                    flag = False
                    break
            if flag:
                break

    indiv = Solution(rand)
    eval(indiv)
    return indiv


# genetic algorithm
def ga(filename, pop):

    p = pop
    selection_op = Tournament()
    with open(filename) as f:
        line = f.readline()
    print line, " ", len(line)
    population = init_individuals(line[:81], p)

    current_best = population[0]
    generation = 0

    while evals < budget:
        #print evals, budget
        nextgeneration = population[:2]
        best = population[0]

        while len(nextgeneration) < p:

            parent_a = selection_op.select(population, 2)
            parent_b = selection_op.select(population, 2)

            child_a, child_b = cross_over(parent_a, parent_b)

            if random.random() < 0.1:
                child_a = mutate(child_a)
                child_b = mutate(child_b)

            #if generation < (budget / p) / 2:
            #    if random.random < 0.1:
            #         child_a = mutate(child_a)
            #         child_b = mutate(child_b)
            #else:
            #    if random.random() < 0.2:
            #        child_a = mutate(child_a)
            #        child_b = mutate(child_b)

            nextgeneration.append(child_a)
            nextgeneration.append(child_b)

        population = sorted(nextgeneration, key = lambda x: x.fitness)
        best = population[0]

        if best.fitness < current_best.fitness:
            current_best = best

        print ",".join([str(generation), str(current_best.fitness)])
        generation += 1

    return current_best

def print_sudoku(representation):
    for i in range(9):
        for j in range(9):
            if (j % 3 == 0) and (j != 0):
                print "|", representation[i][j],
            else:
                print representation[i][j],
        print " "
        if (i % 3 == 2) and (i != 8):
            print "------+------+------"


def usage():
    print 'usage : sudoku.py test.txt -f [budget] -p [population]'
    print 'Please retry'

if __name__ == "__main__":
    if len(sys.argv) != 6:
        usage()
        exit()

    filename = sys.argv[1]
    budget = int(sys.argv[3])
    pop = int(sys.argv[5])

    best = ga(filename, pop)
    print_sudoku(best.representation)
    print "fitness: ", best.fitness
    if best.fitness != 0:
        print "I cannot solve it :("
    else:
        print "Success!"
