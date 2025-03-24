import time

s=time.time()
for i in range(20000): 
  pass
e=time.time()
print("Empty loop time: ", e-s)

