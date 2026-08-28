sweeps={
 "a1_over_a2":[.5,.75,1,1.33,2],
 "sigma":[.05,.10,.20,.40],
 "D":[1e-3,1e-2,1e-1],
 "v":[.5,1,2],
 "dt":[.05,.02,.01,.005],
 "N_MC":[1000,10000,100000],
 "n_spatial":[50,100,200,400],
}

print("Recommended staged sweep:")
print("1) dt, N_MC, n_spatial convergence")
print("2) a1/a2")
print("3) DeltaU/SIGMA")
print("4) sigma, D, v")
print("5) RL memory b")
for k,v in sweeps.items(): print(f"{k:15s}: {v}")
