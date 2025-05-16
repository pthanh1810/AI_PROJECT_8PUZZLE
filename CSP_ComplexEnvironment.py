import pygame
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import time
import random
import math
import heapq

# Khởi tạo Pygame
pygame.init()

# Cấu hình cửa sổ Pygame
CELL_SIZE = 100
BOARD_SIZE = 3 * CELL_SIZE
WINDOW = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
pygame.display.set_caption("8-Puzzle Solver")

# Bảng màu (tông màu xanh)
PRIMARY_BLUE = (30, 144, 255)  # DodgerBlue
DARK_BLUE = (25, 25, 112)      # MidnightBlue
LIGHT_BLUE = (173, 216, 230)   # LightBlue
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
SHADOW = (50, 50, 50, 50)      # Hiệu ứng bóng
LIGHT_BLUE_HEX = "#ccf2ff"     # Nền xanh nhạt
DARK_BLUE_HEX = "#004080"      # Xanh đậm cho nút
WHITE_HEX = "#ffffff"

# Font chữ
FONT = pygame.font.SysFont("Arial", 48, bold=True)

# Trạng thái mục tiêu
GOAL_STATE = [1, 2, 3, 4, 5, 6, 7, 8, 0]

# Các hàm hỗ trợ
def is_goal(state):
    return state == GOAL_STATE

def find_blank(state):
    return state.index(0)

def is_solvable(state):
    inversions = 0
    state_no_zero = [x for x in state if x != 0]
    for i in range(len(state_no_zero)):
        for j in range(i + 1, len(state_no_zero)):
            if state_no_zero[i] > state_no_zero[j]:
                inversions += 1
    return inversions % 2 == 0

def get_next_states(state):
    blank_pos = find_blank(state)
    next_states = []
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    row, col = divmod(blank_pos, 3)
    
    for move_row, move_col in moves:
        new_row, new_col = row + move_row, col + move_col
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_pos = new_row * 3 + new_col
            new_state = state.copy()
            new_state[blank_pos], new_state[new_pos] = new_state[new_pos], new_state[blank_pos]
            next_states.append((new_state, blank_pos, new_pos))
    
    return next_states

def h_manhattan(state):
    total = 0
    for i in range(9):
        if state[i] == 0:
            continue
        target_row, target_col = divmod(state[i] - 1, 3) if state[i] != 0 else (2, 2)
        current_row, current_col = divmod(i, 3)
        total += abs(target_row - current_row) + abs(target_col - current_col)
    return total

# Các thuật toán được giữ lại
def backtracking(start_state, depth_limit=50):
    if not is_solvable(start_state):
        return None, 0, 0
    
    def backtrack(state, path, visited, depth, h_threshold):
        if is_goal(state):
            return path + [state]
        
        if depth >= depth_limit:
            return None
        
        state_tuple = tuple(state)
        if state_tuple in visited:
            return None
        visited.add(state_tuple)
        
        h = h_manhattan(state)
        if h > h_threshold:
            return None
        
        next_states = get_next_states(state)
        next_states.sort(key=lambda x: h_manhattan(x[0]))
        
        new_h_threshold = max(h_threshold - 1, 0) if h > 0 else h_threshold
        
        for next_state, _, _ in next_states:
            result = backtrack(next_state, path + [state], visited, depth + 1, new_h_threshold)
            if result:
                return result
        return None
    
    start_time = time.time()
    visited = set()
    initial_h = h_manhattan(start_state)
    h_threshold = initial_h + 10
    
    if is_goal(start_state):
        return [start_state], 1, time.time() - start_time
    
    result = backtrack(start_state, [], visited, 0, h_threshold)
    return result, len(visited), time.time() - start_time

def forward_checking(start_state, depth_limit=50):
    def is_solvable(state):
        inversions = 0
        state_no_zero = [x for x in state if x != 0]
        for i in range(len(state_no_zero)):
            for j in range(i + 1, len(state_no_zero)):
                if state_no_zero[i] > state_no_zero[j]:
                    inversions += 1
        return inversions % 2 == 0
    
    def backtrack_with_fc(state, path, visited, depth):
        if is_goal(state):
            return path + [state]
        if depth >= depth_limit:
            return None
        
        state_tuple = tuple(state)
        if state_tuple in visited:
            return None
        visited.add(state_tuple)
        
        for next_state, _, _ in get_next_states(state):
            if not is_solvable(next_state):
                continue
            if h_manhattan(next_state) > h_manhattan(state) + 1:
                continue
            result = backtrack_with_fc(next_state, path + [state], visited, depth + 1)
            if result:
                return result
        return None
    
    start_time = time.time()
    visited = set()
    result = backtrack_with_fc(start_state, [], visited, 0)
    return result, len(visited), time.time() - start_time

def min_conflicts(start_state, max_steps=1000):
    def get_conflicts(state):
        return sum(1 for i in range(9) if state[i] != GOAL_STATE[i] and state[i] != 0)
    
    current_state = start_state.copy()
    visited = set()
    visited.add(tuple(current_state))
    path = [current_state]
    start_time = time.time()
    
    for _ in range(max_steps):
        if is_goal(current_state):
            return path, len(visited), time.time() - start_time
        
        next_states = [(state, h_manhattan(state)) for state, _, _ in get_next_states(current_state) if tuple(state) not in visited]
        if not next_states:
            return None, len(visited), time.time() - start_time
        
        next_states.sort(key=lambda x: x[1])
        current_state = next_states[0][0]
        visited.add(tuple(current_state))
        path.append(current_state)
    
    return None, len(visited), time.time() - start_time

def and_or_tree_search(start_state):
    if not is_solvable(start_state):
        return None, 0, 0
    
    def search(state, path, visited, depth_limit, h_threshold):
        if is_goal(state):
            return path + [state], True
        
        if depth_limit <= 0:
            return None, False
        
        state_tuple = tuple(state)
        if state_tuple in visited:
            return None, False
        visited.add(state_tuple)
        
        h = h_manhattan(state)
        if h > h_threshold:
            return None, False
        
        next_states = get_next_states(state)
        next_states.sort(key=lambda x: h_manhattan(x[0]))
        
        new_h_threshold = max(h_threshold - 1, 0) if h > 0 else h_threshold
        
        for next_state, _, _ in next_states:
            result, success = search(next_state, path + [state], visited, depth_limit - 1, new_h_threshold)
            if success:
                return result, True
        
        return None, False
    
    start_time = time.time()
    visited = set()
    depth_limit = 50
    initial_h = h_manhattan(start_state)
    h_threshold = initial_h + 10
    
    if is_goal(start_state):
        return [start_state], 1, time.time() - start_time
    
    result, success = search(start_state, [], visited, depth_limit, h_threshold)
    if success:
        return result, len(visited), time.time() - start_time
    return None, len(visited), time.time() - start_time

def partially_observable_search(start_state, observed_positions=None, max_states_visited=10000, max_time=30):
    if not is_solvable(start_state):
        print("Debug: Trạng thái ban đầu không khả giải")
        return None, 0, 0

    start_time = time.time()
    visited = set()

    if observed_positions is None:
        observed_positions = [0, 1, 3, 4]
    print(f"Debug: Vị trí quan sát: {observed_positions}")

    initial_belief_state = {tuple(start_state)}
    belief_tuple = frozenset(initial_belief_state)
    visited.add(belief_tuple)
    states_visited = 0

    queue = [(h_manhattan(start_state), initial_belief_state, [], start_state)]

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def get_observation(state):
        blank_pos = find_blank(state)
        x, y = divmod(blank_pos, 3)
        observation = {'empty': (x, y), 'adjacent': {}}
        for dx, dy in moves:
            new_x, new_y = x + dx, y + dy
            new_pos = new_x * 3 + new_y
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                observation['adjacent'][(new_x, new_y)] = state[new_pos]
        return observation

    def apply_action(belief_state, action):
        new_belief_state = set()
        dx, dy = action
        for state_tuple in belief_state:
            state = list(state_tuple)
            blank_pos = find_blank(state)
            x, y = divmod(blank_pos, 3)
            new_x, new_y = x + dx, y + dy
            new_pos = new_x * 3 + new_y
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                new_state = state.copy()
                new_state[blank_pos], new_state[new_pos] = new_state[new_pos], new_state[blank_pos]
                new_belief_state.add(tuple(new_state))
        return new_belief_state

    def is_goal_belief(belief_state):
        for state_tuple in belief_state:
            if list(state_tuple) != GOAL_STATE:
                return False
        return True

    while queue and states_visited < max_states_visited and time.time() - start_time < max_time:
        _, current_belief, path, rep_state = heapq.heappop(queue)
        print(f"Debug: Kích thước niềm tin: {len(current_belief)}, Độ dài đường đi: {len(path)}")

        if is_goal_belief(current_belief):
            print("Debug: Đã đạt trạng thái mục tiêu")
            return path + [rep_state], states_visited, time.time() - start_time

        for dx, dy in moves:
            new_belief_state = apply_action(current_belief, (dx, dy))
            if not new_belief_state:
                print(f"Debug: Không có trạng thái mới cho hành động ({dx}, {dy})")
                continue

            filtered_belief_state = set()
            expected_empty_pos = None
            for prev_state_tuple in current_belief:
                prev_state = list(prev_state_tuple)
                blank_pos = find_blank(prev_state)
                x, y = divmod(blank_pos, 3)
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < 3 and 0 <= new_y < 3:
                    expected_empty_pos = (new_x, new_y)
                    break

            for state_tuple in new_belief_state:
                state = list(state_tuple)
                observation = get_observation(state)
                if expected_empty_pos and observation['empty'] == expected_empty_pos:
                    filtered_belief_state.add(state_tuple)

            if filtered_belief_state:
                belief_tuple = frozenset(filtered_belief_state)
                if belief_tuple not in visited:
                    visited.add(belief_tuple)
                    states_visited += len(filtered_belief_state)
                    h_value = sum(h_manhattan(list(s)) for s in filtered_belief_state) / len(filtered_belief_state)
                    new_rep_state = list(next(iter(filtered_belief_state)))
                    heapq.heappush(queue, (h_value, filtered_belief_state, path + [rep_state], new_rep_state))
                    print(f"Debug: Đã thêm trạng thái niềm tin mới, kích thước: {len(filtered_belief_state)}")

    print("Debug: Không tìm thấy lời giải")
    return None, states_visited, time.time() - start_time

# Lớp giao diện PuzzleGUI
class PuzzleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle Solver")
        self.root.configure(bg=LIGHT_BLUE_HEX)
        self.is_running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Cấu hình kiểu dáng
        self.style = ttk.Style()
        self.style.theme_use("default")
        self.style.configure("TButton", font=("Times New Roman", 11, "bold"), padding=6, background=DARK_BLUE_HEX, foreground=WHITE_HEX)
        self.style.map("TButton", background=[("active", "#0059b3")])
        self.style.configure("TLabel", font=("Times New Roman", 12), background=LIGHT_BLUE_HEX)
        self.style.configure("TLabelframe", font=("Times New Roman", 12, "bold"), background=LIGHT_BLUE_HEX)
        self.style.configure("TCheckbutton", font=("Times New Roman", 11), background=LIGHT_BLUE_HEX)

        # Container chính
        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill="both", expand=True)

        # Khung nhập liệu
        self.input_frame = ttk.LabelFrame(self.main_frame, text="NHẬP PUZZLE", padding=10)
        self.input_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        ttk.Label(self.input_frame, text="NHẬP TRẠNG THÁI BAN ĐẦU:").grid(row=0, column=0, columnspan=3, pady=5)
        self.entries = []
        for i in range(3):
            row_entries = []
            for j in range(3):
                entry = ttk.Entry(self.input_frame, width=5, justify="center", font=("Times New Roman", 12))
                entry.grid(row=i+1, column=j, padx=5, pady=5)
                row_entries.append(entry)
            self.entries.append(row_entries)

        # Trạng thái mặc định
        default_state = [2, 6, 5, 8, 0, 7, 4, 3, 1]
        for i in range(3):
            for j in range(3):
                self.entries[i][j].insert(0, str(default_state[i * 3 + j]))

        # Khung chọn thuật toán
        self.algorithm_frame = ttk.LabelFrame(self.main_frame, text="CHỌN THUẬT TOÁN", padding=10)
        self.algorithm_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.algorithm_vars = {}
        algorithms = ["Backtracking", "Forward Checking", "Min-Conflicts", "AND-OR Tree", "Partially Observable"]
        for i, algo in enumerate(algorithms):
            var = tk.BooleanVar(value=False)
            self.algorithm_vars[algo] = var
            ttk.Checkbutton(self.algorithm_frame, text=algo, variable=var).grid(row=i, column=0, sticky="w", padx=5, pady=2)

        # Khung nút
        self.button_frame = ttk.Frame(self.algorithm_frame)
        self.button_frame.grid(row=len(algorithms), column=0, pady=10)
        self.solve_button = ttk.Button(self.button_frame, text="Giải", command=self.solve, style="TButton")
        self.solve_button.grid(row=0, column=0, padx=5)
        self.reset_button = ttk.Button(self.button_frame, text="Đặt lại", command=self.reset, style="TButton")
        self.reset_button.grid(row=0, column=1, padx=5)

        # Khung tốc độ phát
        self.speed_frame = tk.Frame(self.main_frame)
        self.speed_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")
        tk.Label(self.speed_frame, text="Tốc độ phát (giây):", font=("Times New Roman", 11)).pack(side="left")
        self.speed_var = tk.DoubleVar(value=0.5)
        self.speed_scale = tk.Scale(self.speed_frame, from_=0.1, to=2.0, orient="horizontal",
                                    variable=self.speed_var, length=200, resolution=0.1,
                                    font=("Times New Roman", 10))
        self.speed_scale.pack(side="left", padx=10)
        self.speed_label = tk.Label(self.speed_frame, text=f"{self.speed_var.get():.1f}", font=("Times New Roman", 10))
        self.speed_label.pack(side="left")
        self.speed_var.trace("w", self.update_speed_label)

        # Khung hiển thị thời gian
        self.time_frame = ttk.LabelFrame(self.main_frame, text="Thời gian chạy", padding=10)
        self.time_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")
        self.time_label = ttk.Label(self.time_frame, text="Thời gian: N/A", font=("Times New Roman", 12))
        self.time_label.pack()

        # Biến trạng thái
        self.solutions = {}
        self.current_state = None
        self.step = 0
        self.highlight_positions = None
        self.is_auto_playing = False

        # Cấu hình trọng số lưới
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

    def update_speed_label(self, *args):
        self.speed_label.config(text=f"{self.speed_var.get():.1f}")

    def stop_play(self):
        self.is_auto_playing = False

    def on_closing(self):
        self.is_running = False
        self.is_auto_playing = False
        pygame.quit()
        self.root.destroy()

    def get_state_from_input(self):
        state = []
        try:
            for i in range(3):
                for j in range(3):
                    val = int(self.entries[i][j].get())
                    if val < 0 or val > 8:
                        raise ValueError("Số phải từ 0 đến 8!")
                    state.append(val)
            if sorted(state) != list(range(9)):
                raise ValueError("Phải bao gồm tất cả các số từ 0 đến 8!")
        except ValueError as e:
            messagebox.showerror("Lỗi", str(e))
            return None
        return state

    def reset(self):
        for i in range(3):
            for j in range(3):
                self.entries[i][j].delete(0, tk.END)
                self.entries[i][j].insert(0, str([2, 6, 5, 8, 0, 7, 4, 3, 1][i * 3 + j]))
        self.solutions = {}
        self.current_state = None
        self.is_auto_playing = False
        self.time_label.config(text="Thời gian: N/A")
        self.update_board()

    def solve(self):
        state = self.get_state_from_input()
        if state is None:
            return
        
        if not is_solvable(state):
            messagebox.showerror("Lỗi", "Trạng thái không thể giải được!")
            return

        selected_algorithms = [algo for algo, var in self.algorithm_vars.items() if var.get()]
        if not selected_algorithms:
            messagebox.showerror("Lỗi", "Vui lòng chọn ít nhất một thuật toán!")
            return
        
        self.solutions = {}
        self.time_label.config(text="Thời gian: Đang tính toán...")
        
        for algorithm in selected_algorithms:
            print(f"\n=== {algorithm} ===")
            if algorithm == "Backtracking":
                solution, visited_count, runtime = backtracking(state)
            elif algorithm == "Forward Checking":
                solution, visited_count, runtime = forward_checking(state)
            elif algorithm == "Min-Conflicts":
                solution, visited_count, runtime = min_conflicts(state)
            elif algorithm == "AND-OR Tree":
                solution, visited_count, runtime = and_or_tree_search(state)
            elif algorithm == "Partially Observable":
                solution, visited_count, runtime = partially_observable_search(state, observed_positions=[0, 1, 3, 4])
            else:
                continue
            
            self.solutions[algorithm] = (solution, visited_count, runtime)
        
            print(f"Thời gian: {runtime:.4f} giây")
            print(f"Trạng thái đã thăm: {visited_count}")
            if solution:
                print(f"Số bước: {len(solution)-1}")
                print("Đường đi:")
                for i, s in enumerate(solution):
                    print(f"Bước {i}: {s}")
            else:
                print("Không tìm thấy giải pháp.")
            print("-" * 50)
        
        if not any(solution for solution, _, _ in self.solutions.values()):
            self.time_label.config(text="Thời gian: Không tìm thấy giải pháp!")
            messagebox.showinfo("Kết quả", "Không tìm thấy giải pháp cho các thuật toán đã chọn!")
        else:
            selected_algorithm = next((algo for algo, (solution, _, _) in self.solutions.items() if solution), None)
            runtime = self.solutions[selected_algorithm][2] if selected_algorithm else 0
            self.time_label.config(text=f"Thời gian: {runtime:.4f} giây")
            self.auto_play()

    def auto_play(self):
        selected_algorithm = next((algo for algo, (solution, _, _) in self.solutions.items() if solution), None)
        if not selected_algorithm:
            return
        
        self.is_auto_playing = True
        solution = self.solutions[selected_algorithm][0]
        delay = int(self.speed_var.get() * 1000)

        def play_step(step=0):
            if not self.is_auto_playing or not self.is_running or step >= len(solution):
                self.is_auto_playing = False
                return
            
            self.step = step
            self.current_state = solution[step]
            self.highlight_positions = None
            if step > 0:
                next_states = get_next_states(solution[step-1])
                for s, blank_pos, new_pos in next_states:
                    if s == solution[step]:
                        self.highlight_positions = (blank_pos, new_pos)
                        break
            self.update_board()
            self.root.after(delay, play_step, step + 1)

        play_step()

    def update_board(self):
        if not self.is_running:
            return
        WINDOW.fill(LIGHT_BLUE)
        
        shadow_surface = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, SHADOW, (10, 10, BOARD_SIZE-10, BOARD_SIZE-10), border_radius=10)
        WINDOW.blit(shadow_surface, (0, 0))
        
        if self.current_state:
            for i in range(3):
                for j in range(3):
                    num = self.current_state[i * 3 + j]
                    x, y = j * CELL_SIZE + 5, i * CELL_SIZE + 5
                    cell_size = CELL_SIZE - 10
                    color = WHITE
                    border_color = PRIMARY_BLUE
                    border_width = 3
                    
                    if self.highlight_positions:
                        if i * 3 + j == self.highlight_positions[0]:
                            color = PRIMARY_BLUE
                            border_color = DARK_BLUE
                            border_width = 5
                        elif i * 3 + j == self.highlight_positions[1]:
                            color = LIGHT_BLUE
                            border_color = RED
                            border_width = 5
                    
                    pygame.draw.rect(WINDOW, color, (x, y, cell_size, cell_size), border_radius=8)
                    pygame.draw.rect(WINDOW, border_color, (x, y, cell_size, cell_size), border_width, border_radius=8)
                    
                    if num != 0:
                        text = FONT.render(str(num), True, BLACK)
                        text_rect = text.get_rect(center=(x + cell_size // 2, y + cell_size // 2))
                        WINDOW.blit(text, text_rect)
        
        pygame.display.flip()

if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleGUI(root)
    
    def main_loop():
        if app.is_running:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        app.on_closing()
                        return
                root.after(10, main_loop)
            except:
                app.on_closing()
    
    root.after(10, main_loop)
    root.mainloop()