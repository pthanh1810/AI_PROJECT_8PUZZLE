# 8-Puzzle Solver Project
## 1. Mục tiêu
Mục tiêu chính của đồ án là:
- Tạo ra một môi trường giải bài toán 8-puzzle.
- Triển khai và chạy thử các thuật toán tìm kiếm.
- So sánh và đánh giá hiệu quả của các thuật toán thuộc 6 nhóm tìm kiếm phổ biến.
## 2. Mô tả bài toán
Bài toán 8-puzzle là một dạng bài toán trạng thái với các đặc điểm sau:
- **Trạng thái ban đầu (Initial State)**:  
  Là một ma trận 3x3 chứa các số từ 0 đến 8, trong đó `0` đại diện cho ô trống.
- **Tập hợp hành động (Actions)**:  
  Di chuyển ô trống (`0`) theo bốn hướng: Trái, Phải, Lên, Xuống (nếu hợp lệ).
- **Hàm chuyển trạng thái (Transition Function)**:  
  Khi thực hiện một hành động, tạo ra một trạng thái mới bằng cách hoán đổi vị trí ô trống với ô được di chuyển.
- **Trạng thái mục tiêu (Goal State)**:  
  Mục tiêu là sắp xếp các ô theo đúng thứ tự từ 1 đến 8, với ô trống ở góc dưới bên phải. Cụ thể:
## 2.1 Các thuật toán tìm kiếm thông tin
Trong lĩnh vực trí tuệ nhân tạo, một **bài toán tìm kiếm** thường được mô hình hóa dựa trên các thành phần chính sau:
### Cấu trúc của một bài toán tìm kiếm
- **Không gian trạng thái (State Space):**  
  Là tập hợp tất cả các trạng thái có thể xảy ra của bài toán. Mỗi trạng thái đại diện cho một cấu hình hợp lệ trong quá trình giải.
- **Trạng thái khởi đầu (Initial State):**  
  Là trạng thái xuất phát ban đầu, nơi bắt đầu quá trình tìm kiếm.
- **Trạng thái đích (Goal State):**  
  Là trạng thái (hoặc tập hợp các trạng thái) mà bài toán yêu cầu đạt đến. Thuật toán dừng khi tìm thấy trạng thái này.
- **Hàm chuyển đổi (Transition Function):**  
  Là tập các phép biến đổi hợp lệ từ một trạng thái sang các trạng thái kề cận.
- **Hàm kiểm tra đích (Goal Test):**  
  Dùng để kiểm tra xem trạng thái hiện tại có phải là trạng thái đích hay không.
- **Hàm chi phí (Cost Function):**  
  Xác định chi phí để thực hiện một hành động hoặc di chuyển từ trạng thái này sang trạng thái khác (không bắt buộc trong các thuật toán vô định hướng).
### Solution (Giải pháp)
Một **giải pháp** cho bài toán tìm kiếm là một chuỗi các hành động (hoặc các trạng thái trung gian) liên tiếp nhau từ **trạng thái khởi đầu** đến **trạng thái đích**, sao cho đáp ứng đầy đủ các điều kiện ràng buộc của bài toán.  
Giải pháp này là **kết quả đầu ra** mà thuật toán tìm kiếm trả về khi tìm được đường đi tối ưu hoặc hợp lệ.
![Image](https://github.com/user-attachments/assets/48f4cc9e-3d21-49b2-b2b3-aaf21153a468)
![Image](https://github.com/user-attachments/assets/25682b96-c22b-4f3c-a716-0f6996cedf89)
