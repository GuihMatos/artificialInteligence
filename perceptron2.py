# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 23:33:19 2020

@author: ADMIN
"""

"""
Implementacao de perceptron de uma camada
#Perceptron 2
"""

import numpy as np

entradas = np.array([1, 7, 5])
pesos = np.array([0.8, 0.1, 0])

def soma(e, p):
    return e.dot(p) # dot() produto escalar (substitui o for)

s = soma(entradas, pesos)

def stepFunction(soma):
    if soma >= 1:
        return 1
    return 0

r = stepFunction(s)

#print("Resultado soma: ", s)
#print("Resultado stepFunction: ", r)