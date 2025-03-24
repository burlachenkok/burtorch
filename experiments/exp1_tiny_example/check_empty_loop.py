import time

s=time.time()
for i in range(100*10000): 
  pass
e=time.time()
print("Empty loop time: ", e-s)

