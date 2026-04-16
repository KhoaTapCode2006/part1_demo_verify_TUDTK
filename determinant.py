import matrix as mt
import gaussian_eliminate as ge

def determinant(A):
    # Kiểm tra ma trận vuông
    n = len(A)
    m = len(A[0])
    if n != m:
        raise ValueError("Chỉ có thể tính định thức cho ma trận vuông.")

    # Gọi hàm khử Gauss. Truyền None cho vector b
    newMatrix, _, swaps = ge.gaussian_eliminate(A, None)

    # Tính tích các phần tử trên đường chéo chính
    det = 1.0
    for i in range(n):
        det *= newMatrix[i][i]

    # Đổi dấu định thức nếu số lần hoán đổi hàng là số lẻ
    if swaps % 2 != 0:
        det = -det

    return det
