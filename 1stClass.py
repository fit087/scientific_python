# -*- coding: utf-8 -*-
"""
@author: Adolfo Correa
Professor: Iuri Segtovich

"""

print('Hello World')

print ('''
       
Hello

World''')


print ("""
       
Hello

World2""")

x=1
X=25.54e-23

d=8/2
c=8//2
b=7%3

a = 25 + 3.14
a+x


#x = input('Digite alguma coisa ')

x = float(x)

import math as m

print(m.pi)

print(m.log10(10))
print(m.log(m.e))

from math import *

print(pi)


x = 1.253
X = x
print ('\nx = ', x)

u=1/(sqrt(2*pi))*e**(-1/2*x**2)

print ('u = ', u)

y = sin(pi*x)**2

print('y = ', y)

z = e**sin(x)/sqrt(x**2+1)

print('z = ', z, '\n')


print('\nSymbolic\n')


import sympy as sy

x = sy.Symbol('x')

u=1/(sqrt(2*pi))*e**(-1/2*x**2)

print(u)
print(u.subs(x, X))


a = pi*1j

print(a)

c= a+5
print(c)



# --------------------------------------------
carteria1=10
carteria2=10
carteria3=10
garcao=0
restaurante=0

# transação
conta=30


# -------------------------------------------
i = int("1_000_000")
print(i)


# Slicing

list1 = [1,34, 45,6,14,566,25625,14]
print(list1)
print(list1[::-1])

lista_vazia = []
n=8
for i in range(n):
    print("i: ", i)
    lista_vazia += [i/7]

print(lista_vazia)

n = input('numero de elementos ')
try:
    n = int(n)
except:
    print("It has to be a integer")
    raise NameError('It has to be a integer!')
lista = []
somatorio = 0

for i in range(n):
    elem = int(input('1: '))
    lista += [elem]
    somatorio += lista[i]
    
    
soma = sum(lista)
print(soma)
    
    



