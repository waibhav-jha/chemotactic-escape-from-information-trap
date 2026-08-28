memory_bits=[0,1,2,3,4,5,6,8,10]

print("Memory sweep")
for b in memory_bits:
    print(f"b={b:2d}   memory states=2^b={2**b}")

print("For each b: train, freeze policy, evaluate independently, calculate MFPT.")
print("Main result: T_MFPT^*(b) versus b.")
