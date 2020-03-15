# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:12:05 2020

@author: Adolfo Correa
@Professor: Iuri Segtovich
"""

# %% Exercise 1 - Graficos em escala logaritmica
# ============================================================================

print('\nExercise 1 - Graficos em escala logaritmica\n',
      '-------------------------------------------------------------------\n')


def Pvdw(T, V):
    R = 8.314
    a = 2.5
    b = .99e-4
    return R * T / (V - b) - a / V**2


T = 298

import numpy as np

vVol = np.linspace(1e-4, 1e-1, 100)
vVol = np.logspace(-4, -1, 100)

vP = Pvdw(T, vVol)

from matplotlib import pyplot as plt

#plt.plot(vVol, vP)

plt.scatter(vVol, vP)

plt.xscale('log')
plt.xlim(1e-4, 1e-1)
#plt.ylim(0, 1e7)
plt.show



#if __name__ == "__main__":
#    x = read_file()
#    print('x = ', x)

# %% Exercise 2 - Grafico Contourn (as curvas de nivel)
# ============================================================================

print('\nExercise 2 - Grafico Contourn (as curvas de nivel)\n',
      '-------------------------------------------------------------------\n')

vT = np.linspace(300, 1000, 50)

mP = np.zeros((100, 50))

for i in range(100):
    for j in range(50):
        mP[i, j] = Pvdw(vT[j], vVol[i])

plt.figure()
#plt.contourf(vT, vVol, mP)
#plt.contourf(vT, np.log10(vVol), mP)
plt.contourf(vT, np.log10(vVol), np.log10(mP))

# Escala de cores do lado
plt.colorbar()

# %% Exercise 3 - Using meshgrid
# ============================================================================

print('\nExercise 3\n',
      '-------------------------------------------------------------------\n')
mT, mVol = np.meshgrid(vT, vVol)
plt.figure()
mP[mP < 100] = 100
plt.contourf(np.log10(mVol), np.log10(mP), mT)

plt.colorbar()

# %% Exercise 4
# ============================================================================

print('\nExercise 4\n',
      '-------------------------------------------------------------------\n')

# Toolkit for 3D using matplotlib
from mpl_toolkits.mplot3d import Axes3D

# ColorMaps
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


#surf = ax.plot_surface(mT, np.log10(mVol), np.log10(mP))

surf = ax.plot_surface(mT, np.log10(mVol), np.log10(mP),
                       cmap=cm.coolwarm)

# Color Legent
fig.colorbar(surf)

#               10e-3 10e7
ax.scatter(800, -3, 7)

# Erro o ponto esta por cima da curva porem es graficado como se fose que esta 
# por baixo por que o ponto e graficado primeiro e depois eh graficada a curva.

# Mayavi - library for 3D graphics



# %% Exercise 5
# ============================================================================

print('\nExercise 5\n',
      '-------------------------------------------------------------------\n')


def Pvdw(T, V):
    R = 8.314
    a = 2.5
    b = .99e-4
    return R * T / (V - b) - a/V**2


T = 298

import numpy as np

vVol = np.linspace(1e-4, 1e-1, 100)
vVol = np.logspace(-4, -1, 100)

vP = Pvdw(T, vVol)

from matplotlib import pyplot as plt

#plt.plot(vVol, vP)

plt.scatter(vVol, vP)

plt.xscale('log')
plt.xlim(1e-4, 1e-1)
#plt.ylim(0,1e7)
plt.show


Tspec = 298
Pspec = 1e6

print(Pvdw(1e-4, Tspec))


def res(V):
    print('testando V=', V)
    res = Pvdw(V=V, T=Tspec) - Pspec
    print('Res = ', res)
    return res

from scipy import optimize as opt

#           Função a minimizar e chute inicial
ans = opt.root(res, 1e-4)

# ans is lika a dictionary so
vol_SOL = ans['x']  # This return a vector
# Its a vector because it's possible that the input was a vector
# you can see the keys in Variable Explorator

vol_SOL_escalar = ans['x'][0]

# %% Exercise 6 - Isotermas de Luminie
# ============================================================================

print('\nExercise 6\n',
      '-------------------------------------------------------------------\n')


def lang(c, q, k):
    return q*k*c/(1.+k*c)


q = 10
k = 1

import numpy
from matplotlib import pyplot as plt
vc = np.linspace(0, 10, 100)
vl = lang(vc, q, k)
plt.plot(vc, vl)
cexp = [1, 2, 7]
lexp = [1, 5, 9]
plt.scatter(cexp, lexp)


def fobj(par):
    """
        f=\\sum(l_i-l^c(c_i))^2
    """
    q = par[0]
    k = par[1]
    f = 0  # f=\sum(l_i-l^c(c_i))^2
    for i in range(3):
        f += (lexp[i] - lang(cexp[i], q, k))**2
    print('Testando par = ', par)
    print('f = ', f)
    return f


test = fobj([12, 1])

ans = opt.minimize(fobj, [9, 1])

parot = ans['x']

vl2 = lang(vc, parot[0], parot[1])
plt.plot(vc, vl2)

print(test)
print(parot)

# %% Exercise 7 - EDO concentração na reação
# ============================================================================

print('\nExercise 7\n',
      '-------------------------------------------------------------------\n')

# concentrações
ca0 = 1.  # reagente
cb0 = 0.  # produto

k = 100.

# duas equações Diferenciais de integração direta


def dy(y, t):
    # 2 elem 1 dimensão
    dycalc = np.zeros((2,))
    dycalc[0] = -k*y[0]
    dycalc[1] = +2*k*y[0]
    print('sao t= ', t)
    return dycalc

from scipy import integrate as integ

ti = np.linspace(0, .2, 100)
# concentração inicial de reagente e produto
y0 = [ca0, cb0]
ri = integ.odeint(dy, y0, ti)

# reagente e produto plotados
plt.plot(ti, ri)

# %% Exercise 8 - Symbolic Algebra
# ============================================================================

print('\nExercise 8\n',
      '-------------------------------------------------------------------\n')

import sympy as sym

# sym.init_printing(use_latex='mathjax')
sym.init_printing(use_latex='latex')

x = sym.symbols('x')
y = x + 1

expr1 = y.integrate(x)
print(y)
print(expr1)

a = 0
b = 1

# integral definida na mão
expr1.subs(x, b)-expr1.subs(x, a)

y = sym.sin(x)
expr2 = y.diff(x)

y.subs(x, sym.pi)

###

funcao = sym.lambdify(x, expr2)
print(funcao(np.pi))


### Equações

expr3 = 1/x
y = sym.symbols('y')

eq1 = sym.relational.Equality(expr3, y)
eq1.rhs  # right handside
eq1.lhs  # left handside
print(eq1)

# Isolando a variable x
sym.solve(eq1.rhs-eq1.lhs, x)

# %% Exercise 9 -
# ============================================================================

print('\nExercise 9\n',
      '-------------------------------------------------------------------\n')
import numpy as np
from scipy import stats

stats.norm.pdf(0,)
vx = np.linspace(-10, 10, 100)
y = stats.norm.pdf(vx, loc=0, scale=1)
plt.scatter(vx, y)

y2 = stats.norm.cdf(vx, loc=0, scale=1)
plt.scatter(vx, y2)

# coordenada de 97% de confiabilidade
y3 = stats.norm.ppf(.97, loc=0, scale=1)

# integral de -inf or to inf
from scipy import integrate as integ

x3 = np.linspace(-10, 0, 1000)
y3 = stats.norm.pdf(x3, loc=0, scale=1)
# integração com trapecios
#              give the points
cdf = integ.trapz(y3, x3)
print(cdf)


# congela outras variables deixando so x
def integrando(x):
    return stats.norm.pdf(x, loc=0, scale=1)


myCDF = integ.quad(integrando, -10, 0)

# variables todas em mayusculas o explorador de
# variables nao eh mostrado no spyder

# %% Exercise 10 -
# ============================================================================

print('\nExercise 10\n',
      '-------------------------------------------------------------------\n')


import numpy as np
import time
from numba import njit


@njit
def Pvdw(T, V):
    R = 8.314
    a = 2.5
    b = .99e-4
    return R * T / (V - b) - a/V**2


nT = 500
nV = 1000
vVol = np.logspace(1e-4, 1e-1, nV)
# vP = Pvdw(T, vVol)
vT = np.linspace(300, 1000, nT)
mP = np.zeros((nV, nT))
start = time.time()


# This is a decorator
# @njit
# Só podem ser chamadas funções
# que sejam @njit
# @njit
def bloco(vT, vVol, mP, nT, nV):
    for i in range(nV):
        for j in range(nT):
            mP[i, j] = Pvdw(vT[j], vVol[i])
    return


bloco(vT, vVol, mP, nT, nV)
print(time.time()-start)
