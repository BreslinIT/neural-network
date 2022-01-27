import random
from math import *

# with open("testData.txt","w") as data:
# 	with open("testTargets.txt","w") as targets:
# 		for i in range(50000):
# 			x = 1-(2*random.random())
# 			y = 1-(2*random.random())
# 			data.write(str(x)+"\n"+str(y)+"\na\n")
# 			targets.write(str(x*y)+"\na\n")

def sigmoid(x):
    sig = 1 / (1 + exp(-x))
    return sig

with open("testData.txt","w") as data:
	with open("testTargets.txt","w") as targets:
		options = [-1.0,0.0,1.0]
		for i in range(50000):
			x = random.random();
			data.write(str(x)+"\n/\n")
			targets.write(str(sigmoid(x))+"\n/\n")