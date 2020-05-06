import numpy
import matplotlib.pyplot as plt
import math

def f(x):
   return math.sin(x)


x1=[]
y1=[]
x2=[]
y2=[]
x11=[]
y11=[]
y22=[]
x22=[]

for x in numpy.arange(0, 2, 0.01):

    for y in numpy.arange(round(x,2), 2, 0.01):

        if(f(x*y)<=f(x)*f(y)):
            x1.append(round(x, 2))
            x11.append(round(y,2))
            y1.append(round(y,2))
            y11.append(round(x,2))
        if (f(x*y)>f(x)*f(y)):
            x2.append(round(x, 2))
            x22.append(round(y, 2))
            y2.append(round(y, 2))
            y22.append(round(x, 2))

plt.plot(x1,y1,'o',color='lightgreen')
plt.plot(x2,y2,'o',color='tomato')
plt.plot(x11,y11,'o',color='lightgreen')
plt.plot(x22,y22,'o',color='tomato')
plt.title("f(x)=sin(x)")
plt.show()






