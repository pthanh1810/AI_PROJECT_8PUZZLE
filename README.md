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
- Tìm kiếm theo chiều rộng (BFS): Mở rộng không gian tìm kiếm theo từng cấp độ.
- Tìm kiếm theo chiều sâu (DFS): Mở rộng sâu nhất có thể dọc theo mỗi nhánh trước khi quay lui.
- Tìm kiếm chi phí đồng nhất (UCS): Mở rộng nút có chi phí đường đi thấp nhất.
- Tìm kiếm sâu dần lặp lại (IDS): Kết hợp ưu điểm về hiệu quả không gian của DFS và tính đầy đủ của BFS.
- **Hàm kiểm tra đích (Goal Test):**  
  Dùng để kiểm tra xem trạng thái hiện tại có phải là trạng thái đích hay không.
- **Hàm chi phí (Cost Function):**  
  Xác định chi phí để thực hiện một hành động hoặc di chuyển từ trạng thái này sang trạng thái khác (không bắt buộc trong các thuật toán vô định hướng).
### Giải pháp
Một **giải pháp** cho bài toán tìm kiếm là một chuỗi các hành động (hoặc các trạng thái trung gian) liên tiếp nhau từ **trạng thái khởi đầu** đến **trạng thái đích**, sao cho đáp ứng đầy đủ các điều kiện ràng buộc của bài toán.  
Giải pháp này là **kết quả đầu ra** mà thuật toán tìm kiếm trả về khi tìm được đường đi tối ưu hoặc hợp lệ.
![Image](https://github.com/user-attachments/assets/48f4cc9e-3d21-49b2-b2b3-aaf21153a468)
## 2.2 Các thuật toán tìm kiếm không có thông tin
- **Trạng thái ban đầu (Initial State):**  
  Là điểm bắt đầu của quá trình tìm kiếm – trạng thái xuất phát của bài toán.
- **Trạng thái đích (Goal State):**  
  Là trạng thái (hoặc tập hợp các trạng thái) mà thuật toán hướng tới – mục tiêu cần đạt được.
- **Hành động (Actions):**  
  Là tập hợp các thao tác hoặc phép biến đổi có thể thực hiện để di chuyển từ trạng thái hiện tại sang trạng thái kế tiếp.
- **Hàm chi phí (Cost Function):**  
  Xác định chi phí hoặc giá trị đánh đổi khi thực hiện một hành động. Được dùng để tìm giải pháp tối ưu nếu bài toán yêu cầu.
- **Hàm kiểm tra trạng thái đích (Goal Test):**  
  Dùng để kiểm tra xem trạng thái hiện tại có phải là trạng thái đích hay không.
### Giải pháp (Solution)
Một **giải pháp** là chuỗi các hành động (hoặc trạng thái trung gian) dẫn từ **trạng thái ban đầu** đến **trạng thái đích**, sao cho thỏa mãn yêu cầu của bài toán tìm kiếm. Đây là kết quả cuối cùng mà thuật toán tìm kiếm trả về.
![Image](https://github.com/user-attachments/assets/25682b96-c22b-4f3c-a716-0f6996cedf89)
## 2.3. Local Search
Local Search là nhóm thuật toán tìm kiếm không quan tâm đến toàn bộ không gian trạng thái, mà chỉ tập trung vào việc cải thiện trạng thái hiện tại. Thường được sử dụng khi không cần lưu vết đường đi, hoặc không gian trạng thái quá lớn để duyệt toàn bộ.
###  Các thành phần chính của bài toán tìm kiếm:
- **Trạng thái ban đầu (Initial State):**  
  Là điểm bắt đầu của thuật toán – trạng thái xuất phát của bài toán.
- **Trạng thái đích (Goal State):**  
  Là mục tiêu cuối cùng cần đạt được trong không gian trạng thái.
- **Hành động (Actions):**  
  Các phép biến đổi cho phép di chuyển từ trạng thái này sang trạng thái khác.
- **Hàm chi phí (Cost Function):**  
  Xác định chi phí cho mỗi bước đi hoặc hành động – thường dùng để tối ưu.
- **Hàm đánh giá (Heuristic Function):**  
  Ước lượng mức độ tốt của trạng thái hiện tại so với mục tiêu. Hàm này đóng vai trò quan trọng trong các thuật toán tìm kiếm cục bộ như Hill Climbing, Simulated Annealing,...
### Solution (Giải pháp):
Là chuỗi các hành động hoặc trạng thái dẫn từ **trạng thái ban đầu** đến **trạng thái đích**, sao cho thỏa mãn mục tiêu bài toán. Trong Local Search, giải pháp có thể là trạng thái "tốt nhất" đạt được, không nhất thiết phải là tối ưu toàn cục.
![Image](https://github.com/user-attachments/assets/b7416e34-5971-4240-ab22-87d5ffc9a214)
## 2.4. Constraint Satisfaction Problems
Constraint Satisfaction Problems (CSP) là các bài toán trong đó mục tiêu là tìm một sự phân bổ giá trị cho các biến sao cho tất cả các ràng buộc giữa các biến đều được thỏa mãn. CSP là một lĩnh vực quan trọng trong trí tuệ nhân tạo và lý thuyết tối ưu, với ứng dụng trong nhiều bài toán thực tiễn như lập lịch, tìm kiếm, và lập trình ràng buộc.
![Image](https://github.com/user-attachments/assets/4a463e48-7e59-42e8-9235-062f7e2a53c7)
## 2.5. Searching in Complex Environments
Local Search là nhóm thuật toán tìm kiếm không quan tâm đến toàn bộ không gian trạng thái, mà chỉ tập trung vào việc cải thiện trạng thái hiện tại. Thường được sử dụng khi không cần lưu vết đường đi, hoặc không gian trạng thái quá lớn để duyệt toàn bộ.
###  Các thành phần chính của bài toán tìm kiếm:
- **Không gian trạng thái (State space):**  
  Tập hợp tất cả các trạng thái có thể xảy ra trong môi trường, mô tả các cấu hình hoặc vị trí khác nhau của hệ thống.
- **Trạng thái ban đầu (Initial state):**  
  Trạng thái xuất phát, nơi quá trình tìm kiếm bắt đầu.
Trạng thái đích hoặc mục tiêu (Goal state): Trạng thái hoặc tập hợp trạng thái mà ta muốn đạt tới.
- Toán tử chuyển trạng thái (Actions/Operators): Các phép biến đổi cho phép chuyển từ trạng thái này sang trạng thái khác trong không gian trạng thái.
- Hàm kiểm tra trạng thái đích (Goal test): Hàm xác định xem trạng thái hiện tại có phải là trạng thái mục tiêu hay không.
- Thông tin quan sát (Observability): Trong môi trường phức tạp, có thể trạng thái không được quan sát đầy đủ hoặc chỉ quan sát một phần, dẫn đến các bài toán tìm kiếm một phần quan sát (partially observable).
- Mô hình môi trường (Model of environment): Mô tả cách trạng thái chuyển đổi dựa trên hành động, có thể là xác định hoặc ngẫu nhiên. Solution là một chuỗi các hành động hoặc kế hoạch (plan) từ trạng thái ban đầu đến trạng thái mục tiêu, sao cho thỏa mãn các ràng buộc của môi trường và đạt được mục tiêu đề ra.
![Image](https://github.com/user-attachments/assets/ec6cfeee-7c15-47d0-a75f-bd036fdbb00b)


