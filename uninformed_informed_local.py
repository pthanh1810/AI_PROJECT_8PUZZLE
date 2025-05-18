import pygame
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
import heapq
import sys
import time
import threading
import random
import math

# Initialize Pygame
pygame.init()

# Configure Pygame window
CELL_SIZE = 100
BOARD_SIZE = 3 * CELL_SIZE
WINDOW = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
pygame.display.set_caption("8-Puzzle Solver")

# Color palette (Blue tone)
PRIMARY_BLUE = (30, 144, 255)  # DodgerBlue
DARK_BLUE = (25, 25, 112)      # MidnightBlue
LIGHT_BLUE = (173, 216, 230)   # LightBlue
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
SHADOW = (50, 50, 50, 50)      # For shadow effect
LIGHT_BLUE_HEX = "#ccf2ff"     # Màu nền xanh nhạt
DARK_BLUE_HEX = "#004080"      # Màu xanh đậm cho các nút
WHITE = "#ffffff"

PRIMARY_BLUE_HEX = "#1E90FF"
DARK_BLUE_HEX = "#191970"
LIGHT_BLUE_HEX = "#ADD8E6"
WHITE_HEX = "#FFFFFF"
GRAY_HEX = "#C8C8C8"
RED_HEX = "#FF0000"

FONT = pygame.font.SysFont("Arial", 48, bold=True)


GOAL_STATE = ("1", "2", "3", "4", "5", "6", "7", "8", "")


def is_goal(state):
    return state == GOAL_STATE

def find_blank(state):
    return state.index("")

def get_next_states(state):
    blank_index = find_blank(state)
    moves = {
        0: [1, 3], 1: [0, 2, 4], 2: [1, 5],
        3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8],
        6: [3, 7], 7: [4, 6, 8], 8: [5, 7]
    }
    next_states = []
    for move in moves.get(blank_index, []):
        new_state = list(state)
        new_state[blank_index], new_state[move] = new_state[move], new_state[blank_index]
        next_states.append((tuple(new_state), blank_index, move))
    return next_states

def manhattan_distance(state):
    distance = 0
    for i, value in enumerate(state):
        if value == "":
            continue
        target_index = GOAL_STATE.index(value)
        x1, y1 = i % 3, i // 3
        x2, y2 = target_index % 3, target_index // 3
        distance += abs(x1 - x2) + abs(y1 - y2)
    return distance

def is_solvable(state):
    state_no_blank = [x for x in state if x != ""]
    inversions = 0
    for i in range(len(state_no_blank)):
        for j in range(i + 1, len(state_no_blank)):
            if int(state_no_blank[i]) > int(state_no_blank[j]):
                inversions += 1
    return inversions % 2 == 0

# ======= Search Algorithms =======
def bfs(start_state):
    start_time = time.time()
    queue = deque([(start_state, [])])
    visited = set([start_state])
    while queue:
        current_state, path = queue.popleft()
        if current_state == GOAL_STATE:
            return path + [current_state], len(visited), time.time() - start_time
        blank_index = find_blank(current_state)
        for next_state, _, _ in get_next_states(current_state):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [current_state]))
    return None, len(visited), time.time() - start_time

def dfs(start_state):
    start_time = time.time()
    stack = [(start_state, [], 0)]
    visited = set([start_state])
    while stack:
        current_state, path, depth = stack.pop()
        if current_state == GOAL_STATE:
            return path + [current_state], len(visited), time.time() - start_time
        if depth >= 50:
            continue
        for next_state, _, _ in reversed(get_next_states(current_state)):
            if next_state not in visited:
                visited.add(next_state)
                stack.append((next_state, path + [current_state], depth + 1))
    return None, len(visited), time.time() - start_time

def ids(start_state):
    start_time = time.time()
    def dfs_limited(state, path, depth, limit):
        if state == GOAL_STATE:
            return path + [state]
        if depth == limit:
            return None
        for next_state, _, _ in get_next_states(state):
            if next_state not in path:
                result = dfs_limited(next_state, path + [state], depth + 1, limit)
                if result:
                    return result
        return None
    limit = 0
    while True:
        result = dfs_limited(start_state, [], 0, limit)
        if result:
            return result, 0, time.time() - start_time
        limit += 1

def ucs(start_state):
    start_time = time.time()
    queue = [(0, start_state, [])]
    visited = set([start_state])
    while queue:
        cost, current_state, path = heapq.heappop(queue)
        if current_state == GOAL_STATE:
            return path + [current_state], len(visited), time.time() - start_time
        for next_state, _, _ in get_next_states(current_state):
            if next_state not in visited:
                visited.add(next_state)
                heapq.heappush(queue, (cost + 1, next_state, path + [current_state]))
    return None, len(visited), time.time() - start_time

def greedy(start_state):
    start_time = time.time()
    queue = [(manhattan_distance(start_state), start_state, [])]
    visited = set([start_state])
    while queue:
        _, current_state, path = heapq.heappop(queue)
        if current_state == GOAL_STATE:
            return path + [current_state], len(visited), time.time() - start_time
        for next_state, _, _ in get_next_states(current_state):
            if next_state not in visited:
                visited.add(next_state)
                heapq.heappush(queue, (manhattan_distance(next_state), next_state, path + [current_state]))
    return None, len(visited), time.time() - start_time

def a_star(start_state):
    start_time = time.time()
    queue = [(manhattan_distance(start_state), 0, start_state, [])]
    visited = set([start_state])
    while queue:
        f_score, g_score, current_state, path = heapq.heappop(queue)
        if current_state == GOAL_STATE:
            return path + [current_state], len(visited), time.time() - start_time
        for next_state, _, _ in get_next_states(current_state):
            if next_state not in visited:
                visited.add(next_state)
                new_g_score = g_score + 1
                new_h_score = manhattan_distance(next_state)
                new_f_score = new_g_score + new_h_score
                heapq.heappush(queue, (new_f_score, new_g_score, next_state, path + [current_state]))
    return None, len(visited), time.time() - start_time

def ida_star(start_state):
    start_time = time.time()
    
    def search(state, g_score, bound, path, visited):
        h_score = manhattan_distance(state)
        f_score = g_score + h_score
        if f_score > bound:
            return None, f_score
        if state == GOAL_STATE:
            return path, f_score
        min_f = float('inf')
        for next_state, _, _ in get_next_states(state):
            if next_state not in visited:
                visited.add(next_state)
                result, next_f = search(next_state, g_score + 1, bound, path + [next_state], visited)
                visited.remove(next_state)
                if result is not None:
                    return result, f_score
                min_f = min(min_f, next_f)
        return None, min_f

    bound = manhattan_distance(start_state)
    while True:
        visited = {start_state}
        result, t = search(start_state, 0, bound, [start_state], visited)
        if result is not None:
            return result, len(visited), time.time() - start_time
        if t == float('inf'):
            return None, len(visited), time.time() - start_time
        bound = t


def steepest_hill_climbing(start_state):
    start_time = time.time()
    current_state = start_state
    path = [current_state]
    visited = set([current_state])
    while current_state != GOAL_STATE:
        best_state = None
        best_heuristic = float('inf')
        for next_state, _, _ in get_next_states(current_state):
            if next_state not in visited:
                h = manhattan_distance(next_state)
                if h < best_heuristic:
                    best_heuristic = h
                    best_state = next_state
        if best_state is None or best_heuristic >= manhattan_distance(current_state):
            return None, len(visited), time.time() - start_time
        current_state = best_state
        visited.add(current_state)
        path.append(current_state)
    return path, len(visited), time.time() - start_time

def stochastic_hill_climbing(start_state):
    start_time = time.time()
    current_state = start_state
    path = [current_state]
    visited = set([current_state])
    while current_state != GOAL_STATE:
        current_heuristic = manhattan_distance(current_state)
        better_neighbors = []
        for next_state, _, _ in get_next_states(current_state):
            if next_state not in visited:
                h = manhattan_distance(next_state)
                if h < current_heuristic:
                    better_neighbors.append(next_state)
        if not better_neighbors:
            return None, len(visited), time.time() - start_time
        current_state = random.choice(better_neighbors)
        visited.add(current_state)
        path.append(current_state)
    return path, len(visited), time.time() - start_time

def simple_hill_climbing(start_state):
    start_time = time.time()
    current_state = start_state
    path = [current_state]
    visited = set([current_state])
    while current_state != GOAL_STATE:
        current_heuristic = manhattan_distance(current_state)
        for next_state, _, _ in get_next_states(current_state):
            if next_state not in visited:
                h = manhattan_distance(next_state)
                if h < current_heuristic:
                    current_state = next_state
                    visited.add(current_state)
                    path.append(current_state)
                    break
        else:
            return None, len(visited), time.time() - start_time
    return path, len(visited), time.time() - start_time

def simulated_annealing(start_state):
    start_time = time.time()
    current_state = start_state
    path = [current_state]
    temperature = 1000.0
    cooling_rate = 0.995
    min_temperature = 0.01
    while current_state != GOAL_STATE and temperature > min_temperature:
        current_cost = manhattan_distance(current_state)
        next_states = get_next_states(current_state)
        if not next_states:
            break
        next_state, _, _ = random.choice(next_states)
        next_cost = manhattan_distance(next_state)
        cost_diff = next_cost - current_cost
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            current_state = next_state
            path.append(current_state)
        temperature *= cooling_rate
    if current_state == GOAL_STATE:
        return path, len(path), time.time() - start_time
    return None, len(path), time.time() - start_time

def beam_search(start_state, beam_width=3):
    start_time = time.time()
    queue = [(manhattan_distance(start_state), start_state, [])]
    visited = set([start_state])
    while queue:
        queue = heapq.nsmallest(beam_width, queue)
        next_queue = []
        for h, current_state, path in queue:
            if current_state == GOAL_STATE:
                return path + [current_state], len(visited), time.time() - start_time
            for next_state, _, _ in get_next_states(current_state):
                if next_state not in visited:
                    visited.add(next_state)
                    new_h = manhattan_distance(next_state)
                    next_queue.append((new_h, next_state, path + [current_state]))
        queue = heapq.nsmallest(beam_width, next_queue)
    return None, len(visited), time.time() - start_time

def genetic_algorithm(start_state, population_size=100, generations=500, mutation_rate=0.1):
    start_time = time.time()
    def generate_individual():
        state = list(start_state)
        moves = []
        for _ in range(random.randint(5, 20)):
            blank_idx = state.index("")
            next_states = get_next_states(state)
            next_state, _, _ = random.choice(next_states)
            state = list(next_state)
            moves.append(tuple(state))
        return moves
    def fitness(individual):
        if not individual:
            return float('inf')
        final_state = individual[-1]
        return manhattan_distance(final_state)
    def crossover(parent1, parent2):
        if not parent1 or not parent2:
            return parent1 if parent1 else parent2
        split = random.randint(0, min(len(parent1), len(parent2)) - 1)
        child = parent1[:split] + parent2[split:]
        return child
    def mutate(individual):
        if random.random() < mutation_rate and individual:
            idx = random.randint(0, len(individual) - 1)
            state = list(individual[idx])
            blank_idx = state.index("")
            next_states = get_next_states(state)
            next_state, _, _ = random.choice(next_states)
            individual[idx:] = [tuple(next_state)]
        return individual
    population = [generate_individual() for _ in range(population_size)]
    visited = set([start_state])
    for _ in range(generations):
        population = sorted(population, key=fitness)
        if fitness(population[0]) == 0:
            return population[0], len(visited), time.time() - start_time
        elite_size = population_size // 10
        new_population = population[:elite_size]
        while len(new_population) < population_size:
            parent1 = random.choice(population[:population_size // 2])
            parent2 = random.choice(population[:population_size // 2])
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
            if child and tuple(child[-1]) not in visited:
                visited.add(tuple(child[-1]))
        population = new_population[:population_size]
    best_individual = min(population, key=fitness)
    if fitness(best_individual) == 0:
        return best_individual, len(visited), time.time() - start_time
    return None, len(visited), time.time() - start_time

# Updated PuzzleGUI class
class PuzzleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle Solver")
        self.root.configure(bg=LIGHT_BLUE_HEX)
        self.is_running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use("default")
        self.style.configure("TButton", font=("Times New Roman", 11, "bold"), padding=6, background=DARK_BLUE_HEX, foreground=WHITE)
        self.style.map("TButton", background=[("active", "#0059b3")])
        self.style.configure("TLabel", font=("Times New Roman", 12), background=LIGHT_BLUE_HEX)
        self.style.configure("TLabelframe", font=("Times New Roman", 12, "bold"), background=LIGHT_BLUE_HEX)
        self.style.configure("TCheckbutton", font=("Times New Roman", 11), background=LIGHT_BLUE_HEX)

        # Main container
        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill="both", expand=True)

        # Input frame
        self.input_frame = ttk.LabelFrame(self.main_frame, text="INPUT PUZZLE", padding=10)
        self.input_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        ttk.Label(self.input_frame, text="ENTER INITIAL STATE:").grid(row=0, column=0, columnspan=3, pady=5)
        self.entries = []
        for i in range(3):
            row_entries = []
            for j in range(3):
                entry = ttk.Entry(self.input_frame, width=5, justify="center", font=("Times New Roman", 12))
                entry.grid(row=i+1, column=j, padx=5, pady=5)
                row_entries.append(entry)
            self.entries.append(row_entries)

        # Default starting state
        default_state = ["2", "6", "5", "8", "", "7", "4", "3", "1"]
        for i in range(3):
            for j in range(3):
                self.entries[i][j].insert(0, default_state[i * 3 + j])

        # Algorithm selection frame
        self.algorithm_frame = ttk.LabelFrame(self.main_frame, text="SELECT ALGORITHMS", padding=10)
        self.algorithm_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.algorithm_vars = {}
        algorithms = ["BFS", "DFS", "IDS", "UCS", "Greedy", "A*", "IDA*", "Simple HC", "Steepest HC", "Stochastic HC", "Simulated Annealing", "Beam Search", "Genetic"]
        for i, algo in enumerate(algorithms):
            var = tk.BooleanVar(value=False)
            self.algorithm_vars[algo] = var
            ttk.Checkbutton(self.algorithm_frame, text=algo, variable=var).grid(row=i, column=0, sticky="w", padx=5, pady=2)

        # Button frame
        self.button_frame = ttk.Frame(self.algorithm_frame)
        self.button_frame.grid(row=len(algorithms), column=0, pady=10)
        self.solve_button = ttk.Button(self.button_frame, text="Solve", command=self.solve, style="TButton")
        self.solve_button.grid(row=0, column=0, padx=5)
        self.reset_button = ttk.Button(self.button_frame, text="Reset", command=self.reset, style="TButton")
        self.reset_button.grid(row=0, column=1, padx=5)

        # Playback speed frame
        self.speed_frame = tk.Frame(self.main_frame)
        self.speed_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")
        tk.Label(self.speed_frame, text="Playback Speed (s):", font=("Times New Roman", 11)).pack(side="left")
        self.speed_var = tk.DoubleVar(value=0.5)
        self.speed_scale = tk.Scale(self.speed_frame, from_=0.1, to=2.0, orient="horizontal",
                                    variable=self.speed_var, length=200, resolution=0.1,
                                    font=("Times New Roman", 10))
        self.speed_scale.pack(side="left", padx=10)
        self.speed_label = tk.Label(self.speed_frame, text=f"{self.speed_var.get():.1f}", font=("Times New Roman", 10))
        self.speed_label.pack(side="left")
        self.speed_var.trace("w", self.update_speed_label)

        # Time display frame
        self.time_frame = ttk.LabelFrame(self.main_frame, text="Thời gian chạy", padding=10)
        self.time_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")
        self.time_label = ttk.Label(self.time_frame, text="Thời gian: N/A", font=("Times New Roman", 12))
        self.time_label.pack()

        # State variables
        self.solutions = {}
        self.current_state = None
        self.step = 0
        self.highlight_positions = None
        self.is_auto_playing = False

        # Configure grid weights
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
                    val = self.entries[i][j].get().strip()
                    if val == "":
                        state.append("")
                    else:
                        val = str(int(val))
                        if val not in {"1", "2", "3", "4", "5", "6", "7", "8"}:
                            raise ValueError("Số phải từ 1 đến 8 hoặc để trống cho ô rỗng!")
                        state.append(val)
            blank_count = state.count("")
            if blank_count != 1:
                raise ValueError("Phải có đúng 1 ô trống!")
            numbers = [x for x in state if x != ""]
            if sorted(numbers) != ["1", "2", "3", "4", "5", "6", "7", "8"]:
                raise ValueError("Phải bao gồm tất cả các số từ 1 đến 8!")
        except ValueError as e:
            messagebox.showerror("Lỗi", str(e))
            return None
        return tuple(state)

    def reset(self):
        for i in range(3):
            for j in range(3):
                self.entries[i][j].delete(0, tk.END)
                self.entries[i][j].insert(0, ["2", "6", "5", "8", "", "7", "4", "3", "1"][i * 3 + j])
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
        self.root.update()
        
        for algorithm in selected_algorithms:
            print(f"\n=== {algorithm} ===")
            if algorithm == "DFS":
                solution, visited_count, runtime = dfs(state)
            elif algorithm == "BFS":
                solution, visited_count, runtime = bfs(state)
            elif algorithm == "IDS":
                solution, visited_count, runtime = ids(state)
            elif algorithm == "UCS":
                solution, visited_count, runtime = ucs(state)
            elif algorithm == "Greedy":
                solution, visited_count, runtime = greedy(state)
            elif algorithm == "A*":
                solution, visited_count, runtime = a_star(state)
            elif algorithm == "IDA*":
                solution, visited_count, runtime = ida_star(state)
            elif algorithm == "Simple HC":
                solution, visited_count, runtime = simple_hill_climbing(state)
            elif algorithm == "Steepest HC":
                solution, visited_count, runtime = steepest_hill_climbing(state)
            elif algorithm == "Stochastic HC":
                solution, visited_count, runtime = stochastic_hill_climbing(state)
            elif algorithm == "Simulated Annealing":
                solution, visited_count, runtime = simulated_annealing(state)
            elif algorithm == "Beam Search":
                solution, visited_count, runtime = beam_search(state)
            elif algorithm == "Genetic":
                solution, visited_count, runtime = genetic_algorithm(state)
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
                    
                    if num != "":
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