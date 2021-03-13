#!/usr/bin/env python
# coding: utf-8

# In[626]:


import math
import numpy as np
from numpy.linalg import inv
import modern_robotics as mr
import matplotlib.pyplot as plt
import pandas as pd
from queue import PriorityQueue as pq


# In[627]:


class SearchAlgorithm():
    def __init__(self):
        edges = r'C:\Users\pwu154371\Desktop\Robotics Assignments\V-REP_scenes\V-REP_scenes\Scene5_example\edges.csv'
        self.df_edges = pd.read_csv(edges)  
        self.numberofedges = len(df_edges)

        nodes = r'C:\Users\pwu154371\Desktop\Robotics Assignments\V-REP_scenes\V-REP_scenes\Scene5_example\nodes.csv'
        self.df_nodes = pd.read_csv(nodes)  
        self.numberofnodes = len(df_nodes)
        self.OPEN = {} # a dict for open node and its' cost
        self.OPEN[1] = 0
        self.CLOSED = [] # list of closed nodes
        self.cost = {} # a dict for edge costs
        self.nbrs = {} # a dict of sets for each node's nbr
        self.parent = [] # a list of lists for n nodes
        self.heuristic_cost_to_go = df_nodes['heuristic-cost-to-go'] # get the heuristic cost to go from csv file
        self.construct_nbrs_cost_dict()
        self.parent = [None] * self.numberofnodes
    
    def construct_nbrs_cost_dict(self):
        for node in range(self.numberofnodes):
            self.nbrs[node+1] = set()

        for i in range(len(self.df_edges)):
            self.cost[(self.df_edges.iloc[i][0],self.df_edges.iloc[i][1])] = self.df_edges.iloc[i][2]
            self.cost[(self.df_edges.iloc[i][1],self.df_edges.iloc[i][0])] = self.df_edges.iloc[i][2]

            self.nbrs[self.df_edges.iloc[i][0]].add(self.df_edges.iloc[i][1])
            self.nbrs[self.df_edges.iloc[i][1]].add(self.df_edges.iloc[i][0])
    def recontruct_path(self):
        path = [12]
        k = 11
        print(self.parent)
        while k > 1:
            prev = self.parent[k-1]
            path.append(prev)
            k = prev
        return path
    def A_start(self, goal):
        # A start algorithm from textbook pseudo code
        past_cost = np.ones(numberofnodes) * np.inf 
        past_cost[0] = 0
        SUCESS = False
        while self.OPEN:
            current = next(iter(self.OPEN))
            del self.OPEN[current]
            self.CLOSED.append(current)
            if current == goal:
                SUCESS = True
                path = self.recontruct_path()
                path.reverse()
                return path
            for nbr in self.nbrs[current]:
                if int(nbr) in self.CLOSED:
                    continue
                tentative_past_cost = past_cost[current-1] + self.cost[current,int(nbr)]
                if tentative_past_cost < past_cost[int(nbr)-1]:
                    past_cost[int(nbr)-1] = tentative_past_cost
                    self.parent[int(nbr)-1] = current
                    est_total_cost = past_cost[int(nbr)-1] + self.heuristic_cost_to_go[int(nbr)-1]
                    self.OPEN[int(nbr)] = est_total_cost
                    self.OPEN = dict(sorted(self.OPEN.items(), key=lambda item: item[1]))
        return False


# In[628]:


answer = SearchAlgorithm()


# In[629]:


lst = answer.A_start(12)


# In[630]:


lst


# In[631]:


import csv

f = open(r'C:\Users\pwu154371\Desktop\Robotics Assignments\V-REP_scenes\V-REP_scenes\Scene5_example\path.csv', 'w')

lst = [lst]

with f:

    writer = csv.writer(f)
    
    for row in lst:
        writer.writerow(row)


# In[ ]:




