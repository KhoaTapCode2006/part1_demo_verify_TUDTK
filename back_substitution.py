import matrix as mt
import gaussian_eliminate as ge

def back_substitution(U, c):
    """
        Giải hệ phương trình tam giác trên Ux = c.
        Xử lý 3 trường hợp: nghiệm duy nhất, vô nghiệm, vô số nghiệm.
    """
    # Xác định số dòng và số cột của ma trận bậc thang U
    n_rows = len(U)
    n_cols = len(U[0])

    # Xác định vị trí các phần tử chốt trên từng dòng
    pivot_col = []
    for i in range(n_rows):
        pc = -1
        for j in range(n_cols):
            # Nếu phát hiện pivot, đánh dấu vị trí cột chứa nó
            if abs(U[i][j]) > mt.Matrix.Zero:
                pc = j
                break
        pivot_col.append(pc)

    # Kiểm tra hệ phương trình có vô nghiệm không
    for i in range(n_rows):
        # Nếu tồn tại dòng có dạng [0 0 0 ... 0 | k] với k != 0, hệ vô nghiệm
        if pivot_col[i] == -1 and abs(c[i]) > mt.Matrix.Zero:
            return 0, None

    # Lưu các biến cơ sở tương ứng với các cột có phần tử chốt.
    pivot_cols_set = {pc for pc in pivot_col if pc != -1}
    # Lưu các biến tự do tương ứng với các cột không có phần tử chốt.
    free_vars = [j for j in range(n_cols) if j not in pivot_cols_set]

    # Thuật toán thế ngược
    def solve_system(c, free_values):
        # Khởi tạo vector nghiệm x với kích thước bằng số cột 
        x = [0.0] * n_cols
        # Gán giá trị ẩn tự do 
        for k, fv in enumerate(free_vars):
            x[fv] = free_values[k]

        # Duyệt ngược vì giá trị ẩn ở dòng dưới sẽ giúp giải ẩn ở dòng trên.
        for i in range(n_rows - 1, -1, -1):
            pc = pivot_col[i]
            # Nếu dòng toàn số 0, bỏ qua 
            if pc == -1: 
                continue   
        
            # x_pc = (c_i - tổng các phần tử đã biết) / hệ số tại pivot
            sigma = sum(U[i][j] * x[j] for j in range(pc + 1, n_cols))
            x[pc] = (c[i] - sigma) / U[i][pc]
        return x

    # Nghiệm duy nhất
    if not free_vars:
        return 1, solve_system(c, [])

    # Vô số nghiệm
    if free_vars:
        # Gán tất cả biến tự do = 0
        particular_sol = solve_system(c, [0.0] * len(free_vars))
        
        # Hệ nghiệm cơ sở (Null space basis)
        basis = []
        for i in range(len(free_vars)):
            # Tạo vector trị tự do: biến thứ i là 1.0, còn lại là 0.0
            free_values = [0.0] * len(free_vars)
        free_values[i] = 1.0
        # Giải Ux = 0 
        zero_vec = [0.0] * n_rows
        vec_basis = solve_system(zero_vec, free_values)
        basis.append(vec_basis)
        
    return -1, (particular_sol, basis)