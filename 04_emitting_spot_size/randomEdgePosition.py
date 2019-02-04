import numpy as np
import datetime
# zero position, estimated
pos0 = 14  # mm

# list of positions to measure
l1 = np.arange(pos0-2, pos0+2+0.1, 0.1) 

l2_1 = np.arange(pos0-2,pos0-3.5-0.25,-0.25)[1:]  # less fine outside
l2_2 = np.arange(pos0+2,pos0+3.5+0.25,0.25)[1:]  # less fine outside

l3_1 = np.arange(pos0-3.5,pos0-6.5-1,-1)[1:]  # less fine outside
l3_2 = np.arange(pos0+3.5,pos0+6.5+1,1)[1:]  # less fine outside

l = np.concatenate((l1,l2_1,l2_2, l3_1,l3_2), axis=0)

np.random.shuffle(l)
print(len(l))
print(l)
date = datetime.date.today()
np.savetxt('{}_randOrder.out'.format(date), l, delimiter=',', fmt='%.2f')   # X is an array
# 


