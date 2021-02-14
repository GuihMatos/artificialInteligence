# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 20:38:44 2017

@author: Jones
"""
"""
# POSSO UTLIZAR ESSE CODIGO COMO BASE DO MEU PiBiC!!!
#
# LeMbRaR DiSsO AcImA ^
"""
import numpy as np
from sklearn import datasets

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1 - sig)

base = datasets.load_breast_cancer() # carrega dataset
entradas = base.data # pega os dados do dadaset como entradas
valoresSaida = base.target # pega os alvos, resultados do dataset como saidas
saidas = np.empty([569, 1], dtype=int) # cria um array vazio
for i in range(569):
    saidas[i] = valoresSaida[i] #preenche o array vazio com valores do dataset

pesos0 = 2*np.random.random((30,5)) - 1 # 30(atributos de entrada), 5(neuronios na camada escondida)
pesos1 = 2*np.random.random((5,1)) - 1 # 5(numero de pesos camada oculta), 1(neuronio de saida - classes)

epocas = 10000
taxaAprendizagem = 0.3
momento = 1

for j in range(epocas): #para adicionar mais camadas, e preciso adicionar mais linhas nesse for 
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print("Erro: ", mediaAbsoluta)
    
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    pesos1Transposta = pesos1.T
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
    
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)
    
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)
    
print("Taxa de acerto: ", 1 - mediaAbsoluta)