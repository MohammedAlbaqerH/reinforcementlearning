import random 
import pygame
import torch.optim as optim
import numpy as np
import torch
#################
#     color     #
#################

black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)
white = (255, 255, 255)
gray = (20, 20, 20)
perpol = (153, 0, 153)

#Dimnation
windowSize = (500, 500)
numberGrid = 30
gridSize = windowSize[0] // numberGrid

#define window
pygame.init()
win = pygame.display.set_mode(windowSize)

def makeSquer(pos, color):
    bock = windowSize[0] // numberGrid
    pygame.draw.rect(win, color, (pos[0]*bock, pos[1]*bock, bock, bock))

def makeSnake(snake, color = green):
    for s in snake:
        makeSquer(s, color)


def make_grid(color = gray):
    x, y = 0, 0

    for _ in range(numberGrid):
        x = x + gridSize
        y = y + gridSize
        pygame.draw.line(win, color, (x, 0), (x, windowSize[1]))
        pygame.draw.line(win, color, (0, y), (windowSize[0], y))
###################################################################





class Envairoment:
    clock = pygame.time.Clock()

    snake = [(24, 25), (23, 25), (22, 25)]

    food = (random.randint(2, numberGrid - 3), random.randint(2, numberGrid - 3))
    while food in snake:
        food = (random.randint(2, numberGrid - 3), random.randint(2, numberGrid - 3))
    x = snake[0][0]
    y = snake[0][1]
    lifetime = 0
    foodscore = 0
    score = 0
    done = False
    
    def __init__(self):
        pass

    def reset(self):
        self.snake =  [(24, 25), (23, 25), (22, 25)]
        self.x = 24
        self.y = 25
        self.done = False
        self.lifetime = 0
        self.foodscore = 0
        self.score = 0
        self.food = (random.randint(2, numberGrid - 3), random.randint(2, numberGrid - 3))
        while self.food in self.snake:
            self.food = (random.randint(2, numberGrid - 3), random.randint(2, numberGrid - 3))
        self.score = 0
        win.fill(black)
        makeSnake(self.snake)
        makeSquer(self.food, red)
        make_grid()
    
    def just_start(self):
        return self.snake == [(24, 25), (23, 25), (22, 25)]

    def _in_pos_dig(self, point):
        head = self.snake[0]
        b = head[1] - head[0]
        return point[1] == point[0] + b

    def _in_nag_dig(self, point):
        head = self.snake[0]
        b = head[1] + head[0]
        return point[1] == -point[0] + b
    
    def _dist(self, x):
        head = self.snake[0]
        s1 = (head[0] - x[0])**2
        s2 = (head[1] - x[1])**2
        return (s1 + s2)**(1/2)
    
    def _find_something(self, somthing):
        head = self.snake[0]
        
        if somthing[0] == head[0]:
            
            if head[1] > somthing[1]:
                return 0, abs(somthing[1] - head[1])
            if head[1] <= somthing[1]:
                return 1, abs(somthing[1] - head[1])

        if somthing[1] == head[1]:
            if head[0] > somthing[0]:
                return 2, abs(somthing[0] - head[0])
            if head[0] <= somthing[0]:
                return 3, abs(somthing[0] - head[0])

        if self._in_pos_dig(somthing):
            if head[1] > somthing[1]:
                return 4, self._dist(somthing)
            if head[1] <= somthing[1]:
                return 5, self._dist(somthing)
        
        if self._in_nag_dig(somthing):
            if head[1] > somthing[1]:
                return 6, self._dist(somthing)
            if head[1] <= somthing[1]:
                return 7, self._dist(somthing)
        return 0, 0



    def _find_wall(self):
        head = self.snake[0]

        w1 = (head[0], numberGrid)
        w2 = (head[0], 1)
        w3 = (numberGrid, head[1])
        w4 = (0, head[1])
        if (head[0] + head[1] <= numberGrid):
            w5 = (head[0] + head[1] , 0)
            w6 = (0, head[0] + head[1] )
        else:
            w5 = (numberGrid, (head[0] + head[1]) - numberGrid)
            w6 = ((head[0] + head[1]) - numberGrid, numberGrid)
        
        if (head[0] <= head[1]):
            w7 = (0, head[1] - head[0])
            w8 = (numberGrid - (head[1] - head[0]),numberGrid)
        else:
            w7 = (head[0] - head[1],0)
            w8 = (numberGrid, numberGrid - (head[1] - head[0]))
        return w1, w2, w3, w4, w5, w6, w7, w8



        

    def get_state(self):
        
        v = np.zeros((8, 3))
        idx_food, dist_food = self._find_something(self.food)
        idx_tail, dist_tail = self._find_something(self.snake[-1])

        
        v[idx_food] += [dist_food, 0, 0]
        v[idx_tail] += [0, dist_tail, 0]
        for i in self._find_wall():
            idx_wall, dist_wall = self._find_something(i)
            v[idx_wall] += [0, 0, dist_wall]




        if self.done:
            v = np.zeros((8, 3))
        v = torch.tensor(v)
        return v.reshape(1,24)

    def rander(self):
        makeSnake(self.snake)
        makeSquer(self.food, red)
        make_grid()
        
        self.clock.tick(50)
        pygame.display.flip()
    
    def num_action_avilable(self):
        return 4
    
    def take_action(self, action):

        if action == 0:
            self.x += 1
        if action == 1:
            self.x += -1
        if action == 2:
            self.y += -1
        if action == 3:
            self.y += 1

        

        if (self.x, self.y) in self.snake:
            self.done = True
            return self.score
        
        if self.x > 50 or self.y >50 or self.x < 0 or self.y <0:
            self.done = True
            return self.score
        
        self.snake.insert(0, (self.x, self.y))

        if (self.x, self.y) == self.food:
            self.foodscore += 1
            self.lifetime += 1
            self.score = self.foodscore + 0.1*self.lifetime
            while self.food in self.snake:
                self.food = (random.randint(2, numberGrid - 3), random.randint(2, numberGrid - 3))
            return self.score
        else:
            self.lifetime += 1
            laste = self.snake.pop()
            makeSquer(laste, black)
        if self.lifetime == 100:
            self.done = True
            return self.score
        self.score = self.foodscore + 0.1*self.lifetime
        # print(self.score)
        return self.score
        


        
# em = Envairoment()
# scores = []
# import time
# for i in range(1000000):
#     for j in range(1000):
#         em.rander()
#         em.get_state()
        
#         s = em.take_action(random.randint(0, 3))
#         # print(em.get_state())
#         # time.sleep(.00001)
#         if em.done:
#             scores.append(s)
#             em.reset()
            
#             break