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
## 2.1 Uninformed Search Algorithms (Các thuật toán tìm kiếm không thông tin)
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
- DFS (Depth-First Search) có thời gian thực hiện cao nhất (0.1576 giây), chứng tỏ không tối ưu trong bài toán 8-puzzle về mặt thời gian. Mặc dù DFS sử dụng ít bộ nhớ, nhưng dễ đi vào nhánh sai và mất thời gian tìm lối thoát.
- BFS và IDS có thời gian gần như bằng 0, cho thấy rất nhanh trong trường hợp cụ thể này. Tuy nhiên, điều này còn phụ thuộc vào độ sâu của lời giải – nếu trạng thái đích nằm sâu, BFS sẽ tốn nhiều tài nguyên hơn.
- UCS (Uniform-Cost Search) có thời gian thực hiện nhỏ (0.0015 giây), chậm hơn BFS và IDS một chút nhưng đảm bảo tìm giải pháp có chi phí thấp nhất.
- Nhìn chung, DFS kém hiệu quả nhất, còn BFS, IDS và UCS cho kết quả nhanh và phù hợp với bài toán có không gian trạng thái lớn như 8-puzzle, đặc biệt là khi cần giải pháp ngắn hoặc có chi phí tối ưu.
## 2.2 Informed Search Algorithms (Nhóm thuật toán tìm kiếm có thông tin)
- Greedy: Mở rộng nút được ước tính là gần mục tiêu nhất.
- Tìm kiếm A* :Kết hợp chi phí đã đi đến nút và chi phí ước tính đến mục tiêu, đảm bảo tính tối ưu trong các điều kiện nhất định.
- IDA (A sâu dần lặp lại):Một phiên bản sâu dần lặp lại của A*, hữu ích cho các không gian tìm kiếm lớn.
### Giải pháp
Một **giải pháp** là chuỗi các hành động (hoặc trạng thái trung gian) dẫn từ **trạng thái ban đầu** đến **trạng thái đích**, sao cho thỏa mãn yêu cầu của bài toán tìm kiếm. Đây là kết quả cuối cùng mà thuật toán tìm kiếm trả về.
![Image](https://github.com/user-attachments/assets/10029e4f-a1b2-4bc1-b4e4-24d8fefca54f)
![Image](https://github.com/user-attachments/assets/6531a781-87cc-41b0-beb1-1b04c8b4743a)
### Đánh giá thuật toán
- Thuật toán A*: tìm ra lời giải tối ưu một cách hiệu quả, đặc biệt khi sử dụng hàm heuristic thích hợp.
- Greedy : có tốc độ xử lý nhanh hơn nhưng lại không đảm bảo tìm được lời giải tối ưu, dễ đi lạc hướng trong không gian trạng thái phức tạp.
- Thuật toán IDA* : tuy vẫn đảm bảo tính tối ưu, nhưng thời gian thực thi có thể kéo dài hơn.
- Nhìn chung, các thuật toán tìm kiếm có sử dụng heuristic giúp giảm thiểu đáng kể số lượng trạng thái cần xem xét, mang lại hiệu quả vượt trội so với các phương pháp tìm kiếm không sử dụng thông tin.
## 2.3. Local Search (Nhóm Thuật Toán Tìm Kiếm Cục Bộ)
Local Search là nhóm thuật toán tìm kiếm không quan tâm đến toàn bộ không gian trạng thái, mà chỉ tập trung vào việc cải thiện trạng thái hiện tại. Thường được sử dụng khi không cần lưu vết đường đi, hoặc không gian trạng thái quá lớn để duyệt toàn bộ.
- Leo đồi đơn giản: Di chuyển đến hàng xóm có giá trị hàm đánh giá tốt nhất.
- Leo đồi ngẫu nhiên: Giới thiệu tính ngẫu nhiên trong việc chọn hàng xóm tiếp theo.
- Leo đồi dốc nhất: Đánh giá tất cả các hàng xóm và di chuyển đến hàng xóm tốt nhất.
- Simulated Annealing: Cho phép di chuyển đến các trạng thái tồi tệ hơn với một xác suất giảm dần theo thời gian, giúp thoát khỏi các cực tiểu cục bộ.
- Genetic Algorithm: Một thuật toán metaheuristic dựa trên quần thể, lấy cảm hứng từ chọn lọc tự nhiên.
- Local Beam Search: Duy trì và cải thiện nhiều giải pháp ứng viên.
### Giải pháp:
Là chuỗi các hành động hoặc trạng thái dẫn từ **trạng thái ban đầu** đến **trạng thái đích**, sao cho thỏa mãn mục tiêu bài toán. Trong Local Search, giải pháp có thể là trạng thái "tốt nhất" đạt được, không nhất thiết phải là tối ưu toàn cục.
![Image](https://github.com/user-attachments/assets/b4614965-31d2-4184-86cb-c6bc68eb7f6a)
![Image](https://github.com/user-attachments/assets/a28ede6e-ddc8-467c-9a79-1340fd614309)
### Đánh giá thuật toán
Các thuật toán Local Search đơn giản chạy nhanh nhưng dễ bị kẹt ở nghiệm chưa tối ưu. 
- Simulated Annealing: khắc phục nhược điểm này bằng cách cho phép thử nghiệm kém hơn để thoát khỏi bế tắc, phù hợp với bài toán phức tạp.
- Beam Search cân đối tốt giữa tốc độ và bộ nhớ nếu chọn độ rộng (beam width) hợp lý.
- Genetic Algorithm có thể tìm lời giải tốt trong không gian lớn nhưng cần nhiều tính toán và điều chỉnh tham số.
- Nhìn chung , các phương pháp tìm kiếm cục bộ và tiến hóa giúp giải nhanh hơn so với tìm kiếm toàn diện, nhưng không luôn đảm bảo tối ưu tuyệt đối
## 2.4. Constraint Satisfaction Problems (Nhóm Thuật Toán Bài Toán Thỏa Mãn Ràng Buộc)
Constraint Satisfaction Problems (CSP) là các bài toán trong đó mục tiêu là tìm một sự phân bổ giá trị cho các biến sao cho tất cả các ràng buộc giữa các biến đều được thỏa mãn. CSP là một lĩnh vực quan trọng trong trí tuệ nhân tạo và lý thuyết tối ưu, với ứng dụng trong nhiều bài toán thực tiễn như lập lịch, tìm kiếm, và lập trình ràng buộc.
- Backtracking: Một thuật toán tổng quát để tìm tất cả (hoặc một số) giải pháp cho một số bài toán tính toán, xây dựng dần các ứng viên cho giải pháp và từ bỏ một ứng viên ("quay lui") ngay khi xác định rằng ứng viên này không thể hoàn thành thành một giải pháp hợp lệ.
- Backtracking Forward: Một biến thể của quay lui kết hợp kiểm tra phía trước để cắt tỉa không gian tìm kiếm sớm hơn.
- Min-Conflicts: Một thuật toán tìm kiếm cục bộ được thiết kế đặc biệt cho các bài toán thỏa mãn ràng buộc.
![Image](https://github.com/user-attachments/assets/4bcb782c-3bb8-4ca2-bf4b-a8f18060bd31)
![Image](https://github.com/user-attachments/assets/3f72258e-ab5e-4dec-a08d-e99f6bb2e0aa)
### Đánh giá thuật toán
- Backtracking:Là phương pháp cơ bản và dễ triển khai, Backtracking hoạt động bằng cách thử từng giá trị cho biến và quay lui khi gặp xung đột. Tuy đảm bảo tìm được lời giải nếu tồn tại, nhưng hiệu suất thấp khi không gian trạng thái lớn, do phải duyệt qua nhiều khả năng mà không có cơ chế loại bỏ sớm các lựa chọn không hợp lệ.
- Backtracking Forward :cải thiện hiệu suất so với Backtracking bằng cách loại bỏ sớm các giá trị không hợp lệ khỏi miền giá trị của các biến chưa gán, giúp giảm số lần quay lui. Tuy nhiên, chi phí tính toán cao hơn do cần cập nhật miền giá trị sau mỗi lần gán biến, và không phát hiện được tất cả các xung đột tiềm ẩn như các phương pháp kiểm tra tính nhất quán mạnh hơn.
- Min-Conflicts: có thể tìm lời giải nhanh chóng nếu bắt đầu từ một trạng thái khởi đầu tốt. Tuy vậy, nó không đảm bảo tìm được lời giải trong mọi trường hợp, đặc biệt nếu không gian trạng thái có nhiều cực trị cục bộ hoặc trạng thái khởi đầu không tốt.
## 2.5. Searching in Complex Environments (Nhóm thuật toán điều hướng trong môi trường phức tạp)
Trong các bài toán tìm kiếm phức tạp, thường có một số thành phần cơ bản như sau:
- Không gian trạng thái: Tập hợp tất cả các trạng thái có thể có của hệ thống, phản ánh các cấu hình hay vị trí khác nhau trong môi trường.
- Trạng thái khởi đầu: Nơi bắt đầu quá trình tìm kiếm.
- Trạng thái mục tiêu: Trạng thái (hoặc một nhóm trạng thái) mà thuật toán hướng tới.
- Tập hành động hoặc toán tử chuyển trạng thái: Các hành động cho phép chuyển từ trạng thái hiện tại sang trạng thái kế tiếp trong không gian trạng thái.
- Hàm kiểm tra mục tiêu: Dùng để xác định liệu trạng thái hiện tại có phải là trạng thái đích hay không.
- Mức độ quan sát: Trong một số môi trường, trạng thái có thể chỉ được quan sát một phần, tạo thành các bài toán với khả năng quan sát không đầy đủ.
- Mô hình môi trường: Mô tả cách các hành động ảnh hưởng đến trạng thái – có thể xác định (deterministic) hoặc ngẫu nhiên (stochastic).
## Giải pháp:
Là một chuỗi hành động dẫn từ trạng thái ban đầu đến mục tiêu, đảm bảo tuân thủ các ràng buộc và đạt được mục tiêu mong muốn.
![Image](https://github.com/user-attachments/assets/61ffd005-8874-4cf5-b091-7381b9e3de28)
![Image](https://github.com/user-attachments/assets/2437c8a0-d5fa-46e0-aae0-1ea20295b007)
![Image](https://github.com/user-attachments/assets/5655bdab-f504-4d5f-b2d9-a50ee8d59f91)
### Đánh giá thuật toán
- Thuật toán AND-OR Tree có thời gian chạy nhanh nhất, cho thấy khả năng xử lý hiệu quả khi môi trường có thể xác định rõ ràng. Sensorless, hoạt động trong môi trường không có thông tin cảm biến, có thời gian chạy cao hơn do phải duy trì và cập nhật tập hợp các trạng thái khả dĩ liên tục. Trong khi đó, Partially Observable có thời gian chạy chậm nhất vì phải xử lý các phép ước lượng trạng thái và cập nhật belief state qua mỗi bước đi.Nhìn chung, hiệu suất của các thuật toán này trong 8-Puzzle ở quy mô đầy đủ là khá hạn chế, và khó có thể áp dụng trực tiếp mà không sử dụng thêm các kỹ thuật giảm không gian trạng thái, heuristic mạnh hoặc tối ưu hóa hiệu năng. Điều này phản ánh rõ đặc trưng của các môi trường phức tạp và vai trò thiết yếu của việc lựa chọn thuật toán phù hợp với đặc điểm của bài toán.
## 2.6. Introduction to Reinforcement Learning (Nhóm Thuật Toán Học Tăng Cường)
Trong Reinforcement Learning, giải pháp là một chính sách tối ưu là một hàm ánh xạ từ trạng thái đến hành động sao cho tổng phần thưởng tích lũy theo thời gian được tối đa hóa.
![Image](https://github.com/user-attachments/assets/39c8e046-2812-4881-bdc7-069608af021c)
### Đánh giá thuật toán
- Thuật toán Q-Learning khi áp dụng cho bài toán 8-Puzzle gặp nhiều khó khăn do không gian trạng thái lớn và quá trình học cần nhiều thời gian để hội tụ. Vì không sử dụng heuristic, Q-Learning thường chậm và kém hiệu quả nếu không có chiến lược khám phá tốt hoặc kỹ thuật hỗ trợ như Deep Q-Learning. Do đó, hiệu suất thấp và khó áp dụng trực tiếp cho 8-Puzzle đầy đủ nếu không tối ưu hóa thêm.
## 3.Kết luận:
Việc áp dụng các thuật toán thuộc sáu nhóm khác nhau đã cho kết quả thành công trong nhiều trường hợp. Tuy nhiên, trong một số tình huống nhất định, một số thuật toán có thể không tìm được lời giải, nguyên nhân do môi trường không ổn định. Điều này có thể xảy ra khi môi trường thay đổi theo thời gian, chứa yếu tố ngẫu nhiên, hoặc khi việc hiển thị và vẽ môi trường trên màn hình không phản ánh chính xác bên trong. Những yếu tố này khiến quá trình tìm kiếm đường đi hoặc giải pháp của thuật toán gặp khó khăn hoặc thậm chí thất bại, mặc dù về lý thuyết thuật toán vẫn đúng và đầy đủ


