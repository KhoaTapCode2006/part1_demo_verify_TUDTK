import matrix as mt
import gaussian_eliminate as ge

def rank_and_basis(A=None):    
    """
    Tìm hạng và cơ sở của các không gian cột, dòng, nghiệm.
    """
    if A is None:
        A = getattr('A', [])
        
    if not A or not A[0]:
        return 0, [], [], []

    m = len(A)       # Số dòng
    n = len(A[0])    # Số cột

    # 1. Gọi hàm khử Gauss
    ref_matrix = ge.gaussian_eliminate(A, None)[0]
    
    # 2. Tìm các cột Pivot
    pivot_cols = []
    for i in range(len(ref_matrix)):
        for j in range(n):
            if abs(ref_matrix[i][j]) > mt.Matrix.Zero: 
                pivot_cols.append(j)
                break
    # 3. Tính hạng ma trận
    rank = len(pivot_cols)
    
    # 4. Đối với ma trận không gian dòng
    row_basis = [ref_matrix[i] for i in range(rank)]

    # 5. Đối với ma trận không gian cột
    col_basis = []
    for j in pivot_cols:
        col = [A[i][j] for i in range(m)]
        col_basis.append(col)

    # Giari tìm không gian nghiệm, thế ngược
    null_basis = []
    free_cols = [j for j in range(n) if j not in pivot_cols]
    
    for free_j in free_cols:
        x = [0.0] * n
        # Đặt một biến tự do là 1, các biến còn lại là 0
        x[free_j] = 1.0  
        
        # Chạy ngược từ dòng pivot cuối lên trên để tìm nghiệm
        for i in range(rank - 1, -1, -1):
            p_col = pivot_cols[i]
            
            sum_known = sum(ref_matrix[i][k] * x[k] for k in range(p_col + 1, n))
            
            # Thu nghiệm từ phương trình R[i][p_col] * x[p_col] + sum_known = 0
            if abs(ref_matrix[i][p_col]) > mt.Matrix.Zero:
                x[p_col] = -sum_known / ref_matrix[i][p_col]
            else:
                x[p_col] = 0.0
            
        null_basis.append(x)

    # Trả về hạng, cơ sở cột, cơ sở dòng, cơ sở nghiệm
    return rank, col_basis, row_basis, null_basis