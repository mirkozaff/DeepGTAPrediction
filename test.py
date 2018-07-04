import os

d = 'd:data/training'
a = [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
print(a)
