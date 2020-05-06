import numpy
import math

# => Załóżmy, że funkcja jest podmultiplikatywna na przedziale [1, nieskonczonosc)
# pokazemy, ze a>1,755069... . Weźmy x,y należące do przedzialu [1, nieskonczonosc)


def f(x, a):
    return math.log(x + a)

a0=[]
list = []
for a in numpy.arange(1, 3, 0.00001):
    for x in numpy.arange(1, 3, 0.01):
        for y in numpy.arange(x, 3, 0.01):
            if f(x * y, a) > f(y, a) * f(x, a):
                list.append("F")
    if len(list) == 0:
        print(a)
        break
    else:
        list=[]

print(a0)