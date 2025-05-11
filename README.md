# 8-Puzzle Solver Project
##  Mục tiêu
Mục tiêu chính của đồ án là:
- Tạo ra một môi trường giải bài toán 8-puzzle.
- Triển khai và chạy thử các thuật toán tìm kiếm.
- So sánh và đánh giá hiệu quả của các thuật toán thuộc 6 nhóm tìm kiếm phổ biến.
##  Mô tả bài toán
Bài toán 8-puzzle là một dạng bài toán trạng thái với các đặc điểm sau:
- **Trạng thái ban đầu (Initial State)**:  
  Là một ma trận 3x3 chứa các số từ 0 đến 8, trong đó `0` đại diện cho ô trống.
- **Tập hợp hành động (Actions)**:  
  Di chuyển ô trống (`0`) theo bốn hướng: Trái, Phải, Lên, Xuống (nếu hợp lệ).
- **Hàm chuyển trạng thái (Transition Function)**:  
  Khi thực hiện một hành động, tạo ra một trạng thái mới bằng cách hoán đổi vị trí ô trống với ô được di chuyển.
- **Trạng thái mục tiêu (Goal State)**:  
  Mục tiêu là sắp xếp các ô theo đúng thứ tự từ 1 đến 8, với ô trống ở góc dưới bên phải. Cụ thể:

![Image](https://github.com/user-attachments/assets/48f4cc9e-3d21-49b2-b2b3-aaf21153a468)
![Image](https://github.com/user-attachments/assets/25682b96-c22b-4f3c-a716-0f6996cedf89)
