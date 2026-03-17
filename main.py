import numpy as np, time
 
def lu_serial(A):
    n=A.shape[0]; L=np.eye(n); U=A.copy().astype(float)
    for k in range(n-1):
        for i in range(k+1,n):
            if abs(U[k,k])<1e-12: continue
            f=U[i,k]/U[k,k]; L[i,k]=f; U[i,k:]-=f*U[k,k:]
    return L,U
 
def cholesky(A):
    n=A.shape[0]; L=np.zeros_like(A,float)
    for i in range(n):
        for j in range(i+1):
            s=sum(L[i,k]*L[j,k] for k in range(j))
            L[i,j]=(A[i,j]-s)/L[j,j] if i!=j else np.sqrt(max(A[i,i]-s,0))
    return L
 
def benchmark_parallel_structure(sizes=[50,100,200]):
    print("Matrix factorisation benchmarks:")
    for n in sizes:
        A=np.random.rand(n,n); A=A@A.T+n*np.eye(n)
        t=time.time(); L,U=lu_serial(A[:,:n]); t_lu=time.time()-t
        t=time.time(); Lc=cholesky(A); t_ch=time.time()-t
 res=np.linalg.norm(L@U-A[:n,:n])
        print(f"  n={n:3d} | LU: {t_lu:.3f}s (res={res:.1e}) | Cholesky: {t_ch:.3f}s")
 
benchmark_parallel_structure()
print("\nMPI parallelisation: scatter rows across ranks, broadcast pivot row")
