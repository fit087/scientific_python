# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:07:06 2020

@author: Adolfo Correa
Professor: Dr. Iuri Segtovich
"""
import numpy as np
print('\nExercise 1\n')

# Exercise 1
# =============================================
x = [1, 346432, 68, 1223, 5, 47, 678]

max_val = x[0]
for elem in x:
    if elem > max_val:
        max_val = elem
        
print ('The maximum value of the list is\n', max_val)

# -------------------------------------------

max_val = x[0]
position = 0
for i in range(1,len(x)):
    if x[i] > max_val:
        max_val = x[i]
        position = i
        
        
print ('The maximum value of the list is\n', max_val, 'position ', position+1)

# ----------------------------------------

maximo = max(x)
print('max function ', maximo)

print(sum(x))
print(np.size(x))
print(np.shape(x))
print(len(x))

# Exercise 2
# ============================================
print('\nExercise 2\n')
import math as m

y = [0.1, 0.9]
a = [1.2, 1.6]
# k = [[0,0.12], [0.12, 0]]
#     line 1     Line 2
k = [[0,0.12], [0.12, 0]]
n = len(y)
am = 0
for i in range(n):
    for j in range(n):
        am += y[i]*y[j]*m.sqrt(a[i]*a[j])*(1-k[i][j])
        
print('a_m = ', am)

import sympy as sy
sy.init_printing()
kij = sy.MatrixSymbol('k', 2, 2)
yi = sy.MatrixSymbol('y', 2, 1)
ai = sy.MatrixSymbol('a', 2, 1)
#y, a, k = sy.symbols('y a k')
am_sym = yi[i]*yi[j]*sy.sqrt(ai[i]*ai[j])*(1-kij[i,j])
print(am_sym)

# Exercise 3 - serie de Taylor
# ============================================
print('\nExercise 3 - Serie de Taylor\n')

#error = 1
#tol = 1e-6
#n = 8
#while abs(residuo) > tol:
#
#acum = 0
#x = 0.5
#for i in range(n):
#    acum += x**i / m.factorial(i)
#
#residuo = old_result - acum 
#old_result = acum
x, y = 0.5, 0
y=(x**0)/m.factorial(0)
y_ant=y
y+=(x**1)/m.factorial(1)


res = y - y_ant
tol = 1e-6
i = 2
while(abs(res)>tol):
    y_ant = y
    y+=(x**i)/m.factorial(i)
    res=y-y_ant
    i+=1

#n += 1
    
print(y)


# List Comprehension
# ==================================
x = [i for i in range(1,3)]

print(x)

# Exercise 4 - Custom function
# ============================================
print('\nExercise 4 - Funçao\n')

def farentheit(*c):
    c=np.array(c)
    return 1.8*c+32

print(farentheit(37, 100, 0, 21))

# Dictionaries


# Exercise 5 - Secador
# ============================================
print('\nExercise 4 - Secador\n')

def secador(xw_in, xw_out, F=100):
    # Local Variables
    xs_in = 1 - xw_in
    xs_out = 1 - xw_out
    # Computing
    p = F*xs_in/xs_out
    w = F*xw_in-p*xw_out
    r=100*w/F/xw_in
    
    
    return p,w,r

p,w,r = secador(.2,.4)

print(p,w,r)


# Variable Global
global T
def mecher():
    global T
    T=T+273
    return T

# Função recebe outra funcao como argumento
def trapz(f,a,b):
    return (b-a)*(f(a)+f(b))/2
def y(x):
    return (x-1)**2+x*2-3

ans = trapz(f=y,a=0,b=1)
    
# runfile('C:/Users/Aula/Documents/adolfo-correa/2ndClass.py', wdir='C:/Users/Aula/Documents/adolfo-correa')
