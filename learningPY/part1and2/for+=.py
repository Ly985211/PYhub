t=1,2,[30,40]
t[2]+=[50]
"""
1.
Traceback (most recent call last):
  File "d:\PYhub\learningPY\seqs.py", line 2, in <module>
    t[2]+=[50]
TypeError: 'tuple' object does not support item assignment
2.
t=(1, 2, [30, 40, 50])
"""
