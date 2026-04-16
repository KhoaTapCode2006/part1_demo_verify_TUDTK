class Matrix:
    Zero = 1e-15
    
    def __init__(self, matrix, b=None):
        self.A = [list(map(float, row)) for row in matrix]
        self.n = len(self.A)
        self.m = len(self.A[0])
        self.b = [float(x) for x in b] if b is not None else None

    @staticmethod
    def swapRows(mat, i, j):
        mat[i], mat[j] = mat[j], mat[i]

    @staticmethod
    def copyMatrix(mat):
        return [row[:] for row in mat]

    def display(self, mat=None, title="Ma trận"):
        print(f"--- {title} ---")
        target = mat if mat is not None else self.A
        for row in target:
            print([round(x, 4) for x in row])