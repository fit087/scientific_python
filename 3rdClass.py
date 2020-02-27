# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:08:51 2020

@author: Adolfo Correa
"""

# Exercise 1
# ============================================================================

print('\nExercise 1\n',
      '-------------------------------------------------------------------\n')

lista=[1,2,3]

class vectorR3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def __repr__(self):
        return '%s = (%s, %s, %s)' % ('vectorR3', self.x, self.y, self.z)
    
    def prod_interno(self, outro):
        return self.x*outro.x +\
        self.y*outro.y+\
        self.z*outro.z
    
    def __mul__(self, outro):
        return self.prod_interno(outro)
        
        
A = vectorR3(5, 3, 4)
B = vectorR3(10, 3, 2)



print(A)


# Exercise 2
# ============================================================================

print('\nExercise 2\n',
      '-------------------------------------------------------------------\n')

f = open("input.txt", 'r')

line = f.readline()
print(line)
line = f.readline()
print(line)
print(line.split(","))
print(line)

# List Comprehention
x = [float(i) for i in line.split(",")]
print(x)

f.close()



#lines = f.readlines()
#print(lines)

# Exercise 3
# ============================================================================

print('\nExercise 3\n',
      '-------------------------------------------------------------------\n')

import read as r
x = r.read_file()
print('x = ', x)

# Exercise 4
# ============================================================================

print('\nExercise 4\n',
      '-------------------------------------------------------------------\n')

x = 1+1

f = open('output.txt', 'w')

f.write("Results"+'\nx = ')

#fmtsci = '{xxx:.2e} e maior q {yyy:.2f}\n' #scientific format
fmtsci = '{:.2e}\n' #scientific format
fmtdec = '{:.2f}\n' 
fmtint = '{:09d}\n'

f.write(str(x))
#f.write(fmtsci.format(xxx=6.02e23))
f.write(fmtsci.format(6.02e23))
f.write(fmtdec.format(3.1415))

#print(fmtsci.format(xxx=6.02e23, yyy=2))
print(fmtsci.format(6.02e23))
print(fmtdec.format(3.1415))

f.close()


# Exercise 5
# ============================================================================

print('\nExercise 5\n',
      '-------------------------------------------------------------------\n')

print(f'{123456:.2e}')

x = 123456
FMT='.2e'

print(f'o valor de x eh {x:{FMT}}')



# Exercise 6
# ============================================================================

print('\nExercise 6\n',
      '-------------------------------------------------------------------\n')
#       lib            module
from matplotlib import pyplot as plt
from random import randint

T = [randint(0,6) for i in range(7)]
X = [randint(0,6) for i in range(7)]

#plt.figure()
plt.scatter(X,T, marker='*', color='y',label='amostra1')
plt.plot(X, T, ls=':',color='b', label='amostra2')
plt.title("Grafico com pyplot")
plt.ylabel("y-axis")
plt.legend()
plt.xlim(0,6)
plt.savefig("Figure1.png")
plt.show()
#plt.close()


# add_subplot(111)  = 1x1 elemento 1
# 221     = 2x2 elemento 1
# 232     = 2x3 elemento 2


# Exercise 7
# ============================================================================

print('\nExercise 7\n',
      '-------------------------------------------------------------------\n')

fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.scatter(X,T)
erro=[2,2,3,5,6,6,3]
ax.errorbar(X,T,erro, ls='',capsize=5)

fig2.savefig('bigger.png', dpi=1000)

plt.bar(X,T)

for i in range(7):
    plt.text(X[i]+.1, T[i]+2, f'{X[i]:.2f}')

#plt.close()
#ax.close()
#fig2.close()
    
# Exercise 8
# ============================================================================

print('\nExercise 8\n',
      '-------------------------------------------------------------------\n')

import numpy as np

# matrix
M = np.array([[1,2],[3,4]])

print(M.shape)


# Shortcut in spyder ctrl+m

A0 = np.array([1, 5])



A1 = np.array([[1, 2],
          [3, 4],
          [0, 4]])
    

print(A1[0][1])
print(A1[0, 1])
    
# Matrix Multiplication @ ou __matmul__
    
A = np.array([[1, 2, 3],
          [3, 4, 1],
          [3, 3, 2]])

    
B = np.array([[2],
          [3],
          [3]])
    

C = A@B

print(C)
    
# linalg.inv
# array.T

    

#Vetores nao são nem linha nem coluna veja A0 (2,) não é (2,1) ou (1,2)

# shortcut F9 run the line

    
c2_col = np.array([[1],[2],[3]])

c2_line = np.array([[1,2,3]])


# Exercise 9 - Equation System
# ============================================================================

print('\nExercise 9 - Equation System\n',
      '-------------------------------------------------------------------\n')

A = np.array([[1, 1, 1, 1],
          [2, 1, -1, -1],
          [1, -1, -1, 3],
          [-1, 1, 1, 0]],dtype='float')

    
b = np.array([[1],
          [0],
          [4],
          [-2]])
    
    
sol = np.linalg.inv(A)@b

print(sol)

lista = [randint(1,9) for i in range(5)]
print(lista)

array1 = np.array(lista)
array_col = np.reshape(array1, (5,1))
array_line = np.reshape(array1, (1,5))

Vector = np.concatenate((array_col, array_col), axis=0)
matrix = np.concatenate((array_col, array_col), axis=1)


np.savetxt('teste.txt', A)

matrix_readed = np.loadtxt('teste.txt')
print(matrix_readed)



























