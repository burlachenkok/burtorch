#!/usr/bin/env python3

import sys

numbers=[]
for arg in sys.argv[1:]:
   numbers.append(float(arg))

print("input: ", numbers)
print("average: ", sum(numbers)/len(numbers))
print("std.dev: ", (sum([(n - sum(numbers)/len(numbers))**2 for n in numbers])/len(numbers) )**0.5 )
