import matrix as mt
import gaussian_eliminate as ge

def inverse(A): 
    """ Tìm ma trận nghịch đảo bằng phương pháp Gauss-Jordan trên ma trận mở rộng [A | I] """
    n = len(A)
    if n == 0:
        raise ValueError("Ma trận không được để trống")
    if n != len(A[0]):
        raise ValueError("ma trận phải là ma trận vuông")
    
    # tạo ma trận mở rộng [A|I]
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]

    ref_matrix, _, _ = ge.gaussian_eliminate(aug)

    # Duyệt từ dòng cuối lên dòng đầu
    for i in range(n - 1, -1, -1):
        # Phần tử chốt 
        pivot = ref_matrix[i][i]
        
        # Kiểm tra tính khả nghịch bằng mt.Matrix.Zero 
        if abs(pivot) < mt.Matrix.Zero:
            raise ValueError("Ma trận suy biến, không thể tìm ma trận nghịch đảo.")

        # Chia dòng i cho pivot để phần tử chốt bằng 1
        for j in range(i, 2 * n):
            ref_matrix[i][j] /= pivot

        # Khử tất cả các phần tử phía trên phần tử chốt về 0
        for k in range(i - 1, -1, -1):
            factor = ref_matrix[k][i]
            for j in range(i, 2 * n):
                ref_matrix[k][j] -= factor * ref_matrix[i][j]

    # Lấy nửa bên phải của ma trận RREF (A^-1)
    return [row[n:] for row in ref_matrix]