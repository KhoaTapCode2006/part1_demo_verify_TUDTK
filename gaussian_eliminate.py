import matrix as mt

def gaussian_eliminate(A, b=None):
    if not A or not A[0]:
        raise ValueError("Ma trận A không được rỗng.")
    
    n = len(A)
    m = len(A[0])
    
    if b is not None:
        if len(b) != n:
            raise ValueError("Kích thước vector b phải khớp với số hàng của ma trận A.")
        b_new = list(b)  # Tạo bản sao để tránh thay đổi dữ liệu gốc
    else:
        b_new = None

    # Khởi tạo ma trận mới
    newMatrix = mt.Matrix.copyMatrix(A)
    swaps = 0
    row = 0

    for col in range(m):
        if row >= n: 
            break
            
        # Tìm phần tử chốt lớn nhất (dòng có trị tuyệt lớn nhất)
        maxRow = row
        for i in range(row + 1, n):
            if abs(newMatrix[i][col]) > abs(newMatrix[maxRow][col]): 
                maxRow = i

        # Nếu phần tử chốt lớn nhất vẫn gần bằng 0, cột này không thể khử tiếp
        if abs(newMatrix[maxRow][col]) < mt.Matrix.Zero:
            continue

        # Nếu dòng được chọn khác với dòng đang chạy theo m thì hoán đổi 2 dòng đó
        if maxRow != row:
            mt.Matrix.swapRows(newMatrix, maxRow, row)
            if b_new is not None:
                b_new[row], b_new[maxRow] = b_new[maxRow], b_new[row]
            swaps += 1

        # Khử Gauss
        for i in range(row + 1, n):
            # Tính hệ số nhân
            factor = newMatrix[i][col] / newMatrix[row][col]
            
            # Khử trên ma trận A
            for j in range(col, m):
                newMatrix[i][j] -= factor * newMatrix[row][j]
                
            # Khử trên vector b
            if b_new is not None:
                b_new[i] -= factor * b_new[row]
                
        row += 1

    return newMatrix, b_new, swaps