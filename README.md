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
- BFS: Mở rộng không gian tìm kiếm theo từng cấp độ.
- DFS: Mở rộng sâu nhất có thể dọc theo mỗi nhánh trước khi quay lui.
- UCS: Mở rộng nút có chi phí đường đi thấp nhất.
- IDS: Kết hợp ưu điểm về hiệu quả không gian của DFS và tính đầy đủ của BFS.
### Giải pháp
Một **giải pháp** cho bài toán tìm kiếm là một chuỗi các hành động (hoặc các trạng thái trung gian) liên tiếp nhau từ **trạng thái khởi đầu** đến **trạng thái đích**, sao cho đáp ứng đầy đủ các điều kiện ràng buộc của bài toán.  
Giải pháp này là **kết quả đầu ra** mà thuật toán tìm kiếm trả về khi tìm được đường đi tối ưu hoặc hợp lệ.
![Image](https://github.com/user-attachments/assets/66f87ae3-6b5d-45de-a73e-f531d80a11e5)
![Image](https://github.com/user-attachments/assets/5294ef7c-f890-47c4-bf44-dff967b12400)
### Đánh giá thuật toán
-DFS (Depth-First Search) có thời gian thực hiện cao nhất (0.1576 giây), chứng tỏ không tối ưu trong bài toán 8-puzzle về mặt thời gian. Mặc dù DFS sử dụng ít bộ nhớ, nhưng dễ đi vào nhánh sai và mất thời gian tìm lối thoát.
-BFS và IDS có thời gian gần như bằng 0, cho thấy rất nhanh trong trường hợp cụ thể này. Tuy nhiên, điều này còn phụ thuộc vào độ sâu của lời giải – nếu trạng thái đích nằm sâu, BFS sẽ tốn nhiều tài nguyên hơn.
-UCS (Uniform-Cost Search) có thời gian thực hiện nhỏ (0.0015 giây), chậm hơn BFS và IDS một chút nhưng đảm bảo tìm giải pháp có chi phí thấp nhất.
## 2.2 Các thuật toán tìm kiếm không có thông tin
- Greedy: Mở rộng nút được ước tính là gần mục tiêu nhất.
- Tìm kiếm A* :Kết hợp chi phí đã đi đến nút và chi phí ước tính đến mục tiêu, đảm bảo tính tối ưu trong các điều kiện nhất định.
- IDA (A sâu dần lặp lại):Một phiên bản sâu dần lặp lại của A*, hữu ích cho các không gian tìm kiếm lớn.
### Giải pháp
Một **giải pháp** là chuỗi các hành động (hoặc trạng thái trung gian) dẫn từ **trạng thái ban đầu** đến **trạng thái đích**, sao cho thỏa mãn yêu cầu của bài toán tìm kiếm. Đây là kết quả cuối cùng mà thuật toán tìm kiếm trả về.
![Image](https://github.com/user-attachments/assets/10029e4f-a1b2-4bc1-b4e4-24d8fefca54f)
![Image](https://github.com/user-attachments/assets/37dfd420-5565-4716-8abb-8cc53cdf7e2a)
## 2.3. Local Search
Local Search là nhóm thuật toán tìm kiếm không quan tâm đến toàn bộ không gian trạng thái, mà chỉ tập trung vào việc cải thiện trạng thái hiện tại. Thường được sử dụng khi không cần lưu vết đường đi, hoặc không gian trạng thái quá lớn để duyệt toàn bộ.
- Leo đồi đơn giản: Di chuyển đến hàng xóm có giá trị hàm đánh giá tốt nhất.
- Leo đồi ngẫu nhiên: Giới thiệu tính ngẫu nhiên trong việc chọn hàng xóm tiếp theo.
- Leo đồi dốc nhất: Đánh giá tất cả các hàng xóm và di chuyển đến hàng xóm tốt nhất.
- Simulated Annealing: Cho phép di chuyển đến các trạng thái tồi tệ hơn với một xác suất giảm dần theo thời gian, giúp thoát khỏi các cực tiểu cục bộ.
- Genetic Algorithm: Một thuật toán metaheuristic dựa trên quần thể, lấy cảm hứng từ chọn lọc tự nhiên.
- Local Beam Search: Duy trì và cải thiện nhiều giải pháp ứng viên.
###Giải pháp:
Là chuỗi các hành động hoặc trạng thái dẫn từ **trạng thái ban đầu** đến **trạng thái đích**, sao cho thỏa mãn mục tiêu bài toán. Trong Local Search, giải pháp có thể là trạng thái "tốt nhất" đạt được, không nhất thiết phải là tối ưu toàn cục.
![Image](https://github.com/user-attachments/assets/b4614965-31d2-4184-86cb-c6bc68eb7f6a)
![Image](https://github.com/user-attachments/assets/a28ede6e-ddc8-467c-9a79-1340fd614309)
## 2.4. Constraint Satisfaction Problems
Constraint Satisfaction Problems (CSP) là các bài toán trong đó mục tiêu là tìm một sự phân bổ giá trị cho các biến sao cho tất cả các ràng buộc giữa các biến đều được thỏa mãn. CSP là một lĩnh vực quan trọng trong trí tuệ nhân tạo và lý thuyết tối ưu, với ứng dụng trong nhiều bài toán thực tiễn như lập lịch, tìm kiếm, và lập trình ràng buộc.
- Backtracking: Một thuật toán tổng quát để tìm tất cả (hoặc một số) giải pháp cho một số bài toán tính toán, xây dựng dần các ứng viên cho giải pháp và từ bỏ một ứng viên ("quay lui") ngay khi xác định rằng ứng viên này không thể hoàn thành thành một giải pháp hợp lệ.
- Backtracking Forward: Một biến thể của quay lui kết hợp kiểm tra phía trước để cắt tỉa không gian tìm kiếm sớm hơn.
- Min-Conflicts: Một thuật toán tìm kiếm cục bộ được thiết kế đặc biệt cho các bài toán thỏa mãn ràng buộc.
![Image](https://github.com/user-attachments/assets/4bcb782c-3bb8-4ca2-bf4b-a8f18060bd31)
![Image](https://github.com/user-attachments/assets/3f72258e-ab5e-4dec-a08d-e99f6bb2e0aa)
## 2.5. Searching in Complex Environments
Trong các bài toán tìm kiếm phức tạp, thường có một số thành phần cơ bản như sau:
-Không gian trạng thái: Tập hợp tất cả các trạng thái có thể có của hệ thống, phản ánh các cấu hình hay vị trí khác nhau trong môi trường.
-Trạng thái khởi đầu: Nơi bắt đầu quá trình tìm kiếm.
-Trạng thái mục tiêu: Trạng thái (hoặc một nhóm trạng thái) mà thuật toán hướng tới.
-Tập hành động hoặc toán tử chuyển trạng thái: Các hành động cho phép chuyển từ trạng thái hiện tại sang trạng thái kế tiếp trong không gian trạng thái.
-Hàm kiểm tra mục tiêu: Dùng để xác định liệu trạng thái hiện tại có phải là trạng thái đích hay không.
-Mức độ quan sát: Trong một số môi trường, trạng thái có thể chỉ được quan sát một phần, tạo thành các bài toán với khả năng quan sát không đầy đủ.
-Mô hình môi trường: Mô tả cách các hành động ảnh hưởng đến trạng thái – có thể xác định (deterministic) hoặc ngẫu nhiên (stochastic).
-Giải pháp của bài toán là một chuỗi hành động (hoặc một kế hoạch) dẫn từ trạng thái ban đầu đến mục tiêu, đảm bảo tuân thủ các ràng buộc và đạt được mục tiêu mong muốn
![Image](https://github.com/user-attachments/assets/61ffd005-8874-4cf5-b091-7381b9e3de28)
![Image](https://github.com/user-attachments/assets/8ce110f2-e462-4858-b4b7-1a3744c04f38)
## 2.6. Introduction to Reinforcement Learning
Trong Reinforcement Learning, giải pháp là một chính sách tối ưu là một hàm ánh xạ từ trạng thái đến hành động sao cho tổng phần thưởng tích lũy theo thời gian được tối đa hóa.
![Image](https://github.com/user-attachments/assets/39c8e046-2812-4881-bdc7-069608af021c)
## 3.Kết luận:
Việc áp dụng các thuật toán thuộc sáu nhóm khác nhau đã cho kết quả thành công trong nhiều trường hợp. Tuy nhiên, trong một số tình huống nhất định, một số thuật toán có thể không tìm được lời giải, nguyên nhân do môi trường không ổn định. Điều này có thể xảy ra khi môi trường thay đổi theo thời gian, chứa yếu tố ngẫu nhiên, hoặc khi việc hiển thị và vẽ môi trường trên màn hình không phản ánh chính xác bên trong. Những yếu tố này khiến quá trình tìm kiếm đường đi hoặc giải pháp của thuật toán gặp khó khăn hoặc thậm chí thất bại, mặc dù về lý thuyết thuật toán vẫn đúng và đầy đủ


