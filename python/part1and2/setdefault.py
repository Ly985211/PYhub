dict1 = {1:['a1']}
print(dict1) # {1: ['a1']}

dict1[3] = ['c1']
print(dict1) # {1: ['a1'], 3: ['c1']}

key,value = 1, 'a2'
dict1.setdefault(key,[]).append(value)
print(dict1) # {1: ['a1', 'a2'], 3: ['c1']}

key, value = 2 , 'b1'
dict1.setdefault(key,[]).append(value)
print(dict1) # {1: ['a1', 'a2'], 3: ['c1'], 2: ['b1']}



