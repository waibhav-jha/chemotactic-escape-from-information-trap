import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def mfpt_from_transition_matrix(P,absorbing_states):
    N=P.shape[0]
    absorbing=set(absorbing_states)
    transient=np.array([i for i in range(N) if i not in absorbing])
    Q=P[np.ix_(transient,transient)]
    T=spsolve(csr_matrix(np.eye(len(transient))-Q),
              np.ones(len(transient)))
    return T,transient

if __name__=="__main__":
    P=np.array([[.5,.5,0],[0,.5,.5],[0,0,1.]])
    T,states=mfpt_from_transition_matrix(P,[2])
    print("Transient states:",states)
    print("MFPT from state 0:",T[0])
    print("Project: construct P^pi on (x_bin,M).")
