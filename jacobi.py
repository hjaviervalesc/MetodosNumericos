def jacobi(A, b, x0, tol=1.0e-6, maxit=200, verbose=True):
    if verbose:
        print("k\t\tx_k\t\t\terror")
        print(f"{0}\t\t{x0}\t\t")

    xk = x0
    for k in range(1, maxit + 1):
        xkprev = xk.copy()
        ### V1
        xk = xkprev + (b - A @ xkprev) / np.diag(A)
        ### V2
        # for i in range(A.shape[0]):
        #     suma = np.dot(A[i], xkprev)
        #     xk[i] = xkprev[i] + 1 / A[i, i] * (b[i] - suma)
        ### V3
        # for i in range(A.shape[0]):
        #     suma = 0.0
        #     for j in range(A.shape[1]):
        #         suma += A[i, j] * xkprev[j]
        #     xk[i] = xkprev[i] + 1 / A[i, i] * (b[i] - suma)
        error = np.linalg.norm(xk - xkprev, np.inf)
        if verbose:
            print(f"{k}\t\t{xk}\t\t{error:e}")
        if error < tol:
            break
    else:
        print(f"Número máximo de iteraciones {maxit} alcanzado.")
        # No sé si hay convergencia
        xk = None
    return xk


if __name__ == "__main__":
    import numpy as np

    A = np.array([[10, 1], [1, 8]])
    b = np.array([23, 26])
    x0 = np.zeros(len(b))
    jacobi(A, b, x0)
