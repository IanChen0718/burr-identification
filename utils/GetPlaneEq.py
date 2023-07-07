# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:59:52 2021

@author: IanChen
"""
from sympy import Symbol, solve, pprint
# class GetPlaneEq():
# # p1 = [6.6554, -50.9655, 0]
# # p2 = [45.4412, -50.9655, 0]
# # p3 = [6.6559, -53.6968, 15.4897]

# # p1 = [6.6554, 50.9655, 0]
# # p2 = [45.4412, 50.9655, 0]
# # p3 = [6.6559, 53.6968, 15.4897]
#     def __init__(self,p1,p2,p3):
#         self.p1 = p1
#         self.p2 = p2
#         self.p3 = p3
#         self.a = Symbol("a")
#         self.b = Symbol("b")
#         self.c = Symbol("c")
#         self.d = Symbol("d")
#     # Plane Equation: ax+by+cz+d=0
#     def getPanel(self, p1, p2, p3, a, b, c, d):
    
#         a = ( (p2[1]-p1[1])*(p3[2]-p1[2])-(p2[2]-p1[2])*(p3[1]-p1[1]) )
#         b = ( (p2[2]-p1[2])*(p3[0]-p1[0])-(p2[0]-p1[0])*(p3[2]-p1[2]) )
#         c = ( (p2[0]-p1[0])*(p3[1]-p1[1])-(p2[1]-p1[1])*(p3[0]-p1[0]) )
#         d = ( 0-(a*p1[0]+b*p1[1]+c*p1[2]) )
#         return a,b,c,d

#     def GetPara(self):
#         a,b,c,d = self.getPanel(self.p1, self.p2, self.p3, self.a, self.b, self.c, self.d)
#         # print("a =",a)
#         # print("b =",b)
#         # print("c =",c)
#         # print("d =",d)
#         return a,b,c,d

def GetPlaneEq(p1,p2,p3):

    # a = Symbol("a")
    # b = Symbol("b")
    # c = Symbol("c")
    # d = Symbol("d")
    
    a = ( (p2[1]-p1[1])*(p3[2]-p1[2])-(p2[2]-p1[2])*(p3[1]-p1[1]) )
    b = ( (p2[2]-p1[2])*(p3[0]-p1[0])-(p2[0]-p1[0])*(p3[2]-p1[2]) )
    c = ( (p2[0]-p1[0])*(p3[1]-p1[1])-(p2[1]-p1[1])*(p3[0]-p1[0]) )
    d = ( 0-(a*p1[0]+b*p1[1]+c*p1[2]) )
    return a, b, c, d
# p1x = Symbol('p1x')
# p1y = Symbol('p1y')

# p2x = Symbol('p2x')
# p2y = Symbol('p2y')

# ux = Symbol('ux')
# uy = Symbol('uy')

# vx = Symbol('vx')
# vy = Symbol('vy')

# s = Symbol('s')
# t = Symbol('t')

# fx = p1x - p2x + s*ux - t*vx
# fy = p1y - p2y + s*uy - t*vy

# sol = solve((fx, fy), s, t)

# print('s:')
# pprint(sol[s])

# print('t:')
# pprint(sol[t])