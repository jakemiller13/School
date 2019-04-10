#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 06:45:16 2018

@author: Jake
"""

# Setup for Graphs

class Node(object):
    def __init__(self, name):
        '''Assumes name is a string'''
        self.name = name
    def getName(self):
        return self.name
    def __str__(self):
        return self.name

class Edge(object):
    def __init__(self, src, dest):
        '''Assumes src and dest are nodes'''
        self.src = src
        self.dest = dest
    def getSource(self):
        return self.src
    def getDestination(self):
        return self.dest
    def __str__(self):
        return self.src.getName() + '->' + self.dest.getName()

class WeightedEdge(Edge):
    def __init__(self, src, dest, weight = 1.0):
        '''Assumes src and dest are nodes, weight a number'''
        self.src = src
        self.dest = dest
        self.weight = weight
    def getWeight(self):
        return self.weight
    def __str__(self):
        return self.src.getName() + '->(' + str(self.weight) + ')' + self.dest.getName()

class Digraph(object):
    #nodes is a list of the nodes in the graph
    #edges is a dict mapping each node to a list of its children
    def __init__(self):
        self.nodes = []
        self.edges = {}
    def addNode(self, node):
        if node in self.nodes:
            raise ValueError('Duplicate node')
        else:
            self.nodes.append(node)
            self.edges[node] = []
    def addEdge(self, edge):
        src = edge.getSource()
        dest = edge.getDestination()
        if not (src in self.nodes and dest in self.nodes):
            raise ValueError('Node not in graph')
            self.edges[src].append(dest)
    def childrenOf(self, node):
        return self.edges[node]
    def hasNode(self, node):
        return node in self.nodes
    def __str__(self):
        result = ''
        for src in self.nodes:
            for dest in self.edges[src]:
                result = result + src.getName() + '->' + dest.getName() + '\n'
            return result[:-1] #omit final new line

class Graph(Digraph):
    def addEdge(self, edge):
        Digraph.addEdge(self, edge)
        rev = Edge(edge.getDestination(), edge.getSource())
        Digraph.addEdge(self, rev)


# Problem - Add the appropriate edges to the graph

nodes = []  #nodes is a list of nodes
nodes.append(Node("ABC")) # nodes[0]
nodes.append(Node("ACB")) # nodes[1]
nodes.append(Node("BAC")) # nodes[2]
nodes.append(Node("BCA")) # nodes[3]
nodes.append(Node("CAB")) # nodes[4]
nodes.append(Node("CBA")) # nodes[5]

g = Graph()
for n in nodes:
    g.addNode(n)

# Start problem

g.addEdge(Edge(nodes[0],nodes[1]))
g.addEdge(Edge(nodes[0],nodes[2]))
g.addEdge(Edge(nodes[1],nodes[4]))
g.addEdge(Edge(nodes[2],nodes[3]))
g.addEdge(Edge(nodes[3],nodes[5]))
g.addEdge(Edge(nodes[4],nodes[5]))

# Exercise 7 - redo class WeightedEdge

class WeightedEdge(Edge):
    def __init__(self, src, dest, weight = 1.0):
        '''Assumes src and dest are nodes, weight a number'''
        self.src = src
        self.dest = dest
        self.weight = weight
    def getWeight(self):
        return self.weight
    def __str__(self):
        return self.src.getName() + '->' + self.dest.getName() + ' (' + str(self.weight) + ')'
        
# Test

h = Graph()
for n in nodes:
    h.addNode(n)
    
h.addEdge(WeightedEdge(nodes[0],nodes[1],2))
h.addEdge(WeightedEdge(nodes[0],nodes[2],3))
h.addEdge(WeightedEdge(nodes[1],nodes[4],4))
h.addEdge(WeightedEdge(nodes[2],nodes[3],5))
h.addEdge(WeightedEdge(nodes[3],nodes[5],6))
h.addEdge(WeightedEdge(nodes[4],nodes[5],7))