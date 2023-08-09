from array import array
from random import random
floats=array('d',(random() for i in range(10**7)))
print(floats[-1])

fp=open('./learningPY/part1and2/floats2.bin','wb')
floats.tofile(fp)
fp.close()

floats2=array('d')
fp=open('./learningPY/part1and2/floats2.bin','rb')
floats2.fromfile(fp,10**7)
fp.close()

print(floats2==floats)

"""
0.7980881882043565
True
"""