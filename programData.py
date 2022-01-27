import random
import numpy as np

# with open("testData.txt","w") as data:
# 	with open("testTargets.txt","w") as targets:
# 		for i in range(50000):
# 			x = 1-(2*random.random())
# 			y = 1-(2*random.random())
# 			data.write(str(x)+"\n"+str(y)+"\n/\n")
# 			targets.write(str(x*y)+"\n/\n")


with open("testData.txt","w") as data:
	with open("testTargets.txt","w") as targets:
		for i in range(10000):
			x = 1-(2*random.random())
			y = 1-(2*random.random())
			data.write(str(x)+"\n"+str(y)+"\n/\n")
			targets.write(str(0.8/(1+np.e**(2.7+(-5*((np.cosh(x)**2+5))/(12*np.sinh(x)**2)))))+"\n/\n")