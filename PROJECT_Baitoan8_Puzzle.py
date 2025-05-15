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
import uuid

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
# Hex colors for Tkinter
PRIMARY_BLUE_HEX = "#1E90FF"
DARK_BLUE_HEX = "#191970"
LIGHT_BLUE_HEX = "#ADD8E6"
WHITE_HEX = "#FFFFFF"
GRAY_HEX = "#C8C8C8"
RED_HEX = "#FF0000"

# Font
FONT = pygame.font.SysFont("Arial", 48, bold=True)

# Goal state
GOAL_STATE = [1, 2, 3, 4, 5, 6, 7, 8, 0]

# Search algorithms (unchanged, with fixed stochastic_hill_climbing)
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

def h_misplaced(state):
    return sum(1 for i in range(9) if state[i] != GOAL_STATE[i] and state[i] != 0)

def h_manhattan(state):
    total = 0
    for i in range(9):
        if state[i] == 0:
            continue
        target_row, target_col = divmod(state[i] - 1, 3) if state[i] != 0 else (2, 2)
        current_row, current_col = divmod(i, 3)
        total += abs(target_row - current_row) + abs(target_col - current_col)
    return total
def beam_search(start_state, beam_width=3):
    start_time = time.time()
    visited = set()
    visited.add(tuple(start_state))
    beam = [(h_manhattan(start_state), start_state, [])]
    
    while beam:
        next_beam = []
        for h, state, path in beam:
            if is_goal(state):
                return path + [state], len(visited), time.time() - start_time
            
            for next_state, _, _ in get_next_states(state):
                state_tuple = tuple(next_state)
                if state_tuple not in visited:
                    visited.add(state_tuple)
                    h_val = h_manhattan(next_state)
                    next_beam.append((h_val, next_state, path + [state]))
        
        # Sắp xếp theo giá trị heuristic và giữ lại top beam_width trạng thái
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:beam_width]
        
        if not beam:
            break
    
    return None, len(visited), time.time() - start_time

def bfs(start_state):
    queue = deque([(start_state, [])])
    visited = set()
    visited.add(tuple(start_state))
    start_time = time.time()
    
    while queue:
        state, path = queue.popleft()
        
        if is_goal(state):
            end_time = time.time()
            return path + [state], len(visited), end_time - start_time
        
        for next_state, _, _ in get_next_states(state):
            state_tuple = tuple(next_state)
            if state_tuple not in visited:
                visited.add(state_tuple)
                queue.append((next_state, path + [state]))
    
    return None, len(visited), time.time() - start_time

def dfs(start_state):
    stack = [(start_state, [])]
    visited = set()
    visited.add(tuple(start_state))
    start_time = time.time()
    
    while stack:
        state, path = stack.pop()
        
        if is_goal(state):
            end_time = time.time()
            return path + [state], len(visited), end_time - start_time
        
        for next_state, _, _ in get_next_states(state):
            state_tuple = tuple(next_state)
            if state_tuple not in visited:
                visited.add(state_tuple)
                stack.append((next_state, path + [state]))
    
    return None, len(visited), time.time() - start_time
def uniform_cost_search(start_state):
    queue = [(0, start_state, [])]  # (cost, state, path)
    visited = set()
    visited.add(tuple(start_state))
    start_time = time.time()
    
    while queue:
        cost, state, path = heapq.heappop(queue)
        
        if is_goal(state):
            end_time = time.time()
            return path + [state], len(visited), end_time - start_time
        
        for next_state, _, _ in get_next_states(state):
            state_tuple = tuple(next_state)
            if state_tuple not in visited:
                visited.add(state_tuple)
                new_cost = cost + 1  # Each move has a cost of 1
                heapq.heappush(queue, (new_cost, next_state, path + [state]))
    

def greedy(start_state):
    queue = [(h_misplaced(start_state), start_state, [])]
    visited = set()
    visited.add(tuple(start_state))
    start_time = time.time()
    
    while queue:
        _, state, path = heapq.heappop(queue)
        
        if is_goal(state):
            end_time = time.time()
            return path + [state], len(visited), end_time - start_time
        
        for next_state, _, _ in get_next_states(state):
            state_tuple = tuple(next_state)
            if state_tuple not in visited:
                visited.add(state_tuple)
                heapq.heappush(queue, (h_misplaced(next_state), next_state, path + [state]))
    
    return None, len(visited), time.time() - start_time

def a_star(start_state):
    queue = [(0 + h_manhattan(start_state), 0, start_state, [])]
    visited = set()
    visited.add(tuple(start_state))
    start_time = time.time()
    
    while queue:
        _, g, state, path = heapq.heappop(queue)
        
        if is_goal(state):
            end_time = time.time()
            return path + [state], len(visited), end_time - start_time
        
        for next_state, _, _ in get_next_states(state):
            state_tuple = tuple(next_state)
            if state_tuple not in visited:
                visited.add(state_tuple)
                new_g = g + 1
                f = new_g + h_manhattan(next_state)
                heapq.heappush(queue, (f, new_g, next_state, path + [state]))
    
    return None, len(visited), time.time() - start_time

def ida_star(start_state):
    def search(state, g, threshold, path, visited):
        h = h_manhattan(state)
        f = g + h
        if f > threshold:
            return None, f
        if is_goal(state):
            return path + [state], f
        
        min_f = float('inf')
        for next_state, _, _ in get_next_states(state):
            state_tuple = tuple(next_state)
            if state_tuple not in visited:
                visited.add(state_tuple)
                result, new_f = search(next_state, g + 1, threshold, path + [state], visited)
                if result:
                    return result, new_f
                min_f = min(min_f, new_f)
        return None, min_f

    start_time = time.time()
    threshold = h_manhattan(start_state)
    visited = set()
    
    while True:
        visited.clear()
        visited.add(tuple(start_state))
        result, new_threshold = search(start_state, 0, threshold, [], visited)
        if result:
            return result, len(visited), time.time() - start_time
        if new_threshold == float('inf'):
            return None, len(visited), time.time() - start_time
        threshold = new_threshold

def ids(start_state):
    def dls(state, path, depth, visited):
        if depth < 0:
            return None
        if is_goal(state):
            return path + [state]
        for next_state, _, _ in get_next_states(state):
            state_tuple = tuple(next_state)
            if state_tuple not in visited:
                visited.add(state_tuple)
                result = dls(next_state, path + [state], depth - 1, visited)
                if result:
                    return result
        return None

    start_time = time.time()
    depth = 0
    while True:
        visited = set()
        visited.add(tuple(start_state))
        result = dls(start_state, [], depth, visited)
        if result:
            return result, len(visited), time.time() - start_time
        depth += 1

def simple_hill_climbing(start_state):
    current_state = start_state.copy()
    visited = set()
    visited.add(tuple(current_state))
    path = [current_state]
    start_time = time.time()

    while True:
        if is_goal(current_state):
            return path, len(visited), time.time() - start_time

        next_states = get_next_states(current_state)

        next_states.sort(key=lambda x: h_manhattan(x[0]))

        improved = False
        for next_state, _, _ in next_states:
            if tuple(next_state) not in visited and h_manhattan(next_state) < h_manhattan(current_state):
                current_state = next_state
                visited.add(tuple(current_state))
                path.append(current_state)
                improved = True
                break

        if not improved:
            return None, len(visited), time.time() - start_time

def steepest_ascent_hill_climbing(start_state):
    current_state = start_state.copy()
    visited = set()
    visited.add(tuple(current_state))
    path = [current_state]
    start_time = time.time()
    
    while True:
        if is_goal(current_state):
            return path, len(visited), time.time() - start_time
            
        next_states = get_next_states(current_state)
        best_state = current_state
        best_h = h_manhattan(current_state)
        
        for next_state, _, _ in next_states:
            if tuple(next_state) not in visited:
                h = h_manhattan(next_state)
                if h < best_h:
                    best_h = h
                    best_state = next_state
        
        if best_state == current_state:
            return None, len(visited), time.time() - start_time
        
        current_state = best_state
        visited.add(tuple(current_state))
        path.append(current_state)

def stochastic_hill_climbing(start_state):
    current_state = start_state.copy()
    visited = set()
    visited.add(tuple(current_state))
    path = [current_state]
    start_time = time.time()
    
    while True:
        if is_goal(current_state):
            return path, len(visited), time.time() - start_time
            
        next_states = get_next_states(current_state)
        better_neighbors = [
            (state, h_manhattan(state)) 
            for state, _, _ in next_states 
            if tuple(state) not in visited and h_manhattan(state) < h_manhattan(current_state)
        ]
        
        if not better_neighbors:
            return None, len(visited), time.time() - start_time
        
        current_state = random.choice([state for state, _ in better_neighbors])
        visited.add(tuple(current_state))
        path.append(current_state)

def belief_state_search(start_states, goal_state, observed_positions=None):
    start_time = time.time()
    if not all(is_solvable(state) for state in start_states):
        print("Debug: Một hoặc nhiều trạng thái ban đầu không khả giải.")
        return None, 0, time.time() - start_time

    if observed_positions is None:
        observed_positions = [0, 1, 3, 4]
    print(f"Debug: Vị trí quan sát: {observed_positions}")

    observed_values = {i: start_states[0][i] for i in observed_positions}
    valid_start_states = [
        state for state in start_states
        if all(state[i] == observed_values[i] for i in observed_positions)
    ]
    if not valid_start_states:
        print("Debug: Không có trạng thái ban đầu nào nhất quán với các vị trí quan sát.")
        return None, 0, time.time() - start_time
    print(f"Debug: Số trạng thái ban đầu hợp lệ: {len(valid_start_states)}")

    visited = set()

    def is_goal_belief(belief_state):
        return all(list(state) == goal_state for state in belief_state)

    def get_consistent_successors(belief_state):
        successor_beliefs = []
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        rep_state = list(next(iter(belief_state)))
        blank_pos = find_blank(rep_state)
        row, col = divmod(blank_pos, 3)

        for move_row, move_col in moves:
            new_row, new_col = row + move_row, col + move_col
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_pos = new_row * 3 + new_col
                new_belief = set()
                for state in belief_state:
                    state = list(state)
                    state_blank_pos = find_blank(state)
                    state_row, state_col = divmod(state_blank_pos, 3)
                    state_new_row, state_new_col = state_row + move_row, state_new_col + move_col
                    if 0 <= state_new_row < 3 and 0 <= state_new_col < 3:
                        state_new_pos = state_new_row * 3 + state_new_col
                        new_state = state.copy()
                        new_state[state_blank_pos], new_state[state_new_pos] = (
                            new_state[state_new_pos], new_state[state_blank_pos]
                        )
                        is_consistent = all(
                            new_state[i] == observed_values[i] for i in observed_positions
                        )
                        if is_consistent and is_solvable(new_state):
                            new_belief.add(tuple(new_state))
                if new_belief:
                    new_rep_state = rep_state.copy()
                    new_rep_state[blank_pos], new_rep_state[new_pos] = (
                        new_rep_state[new_pos], new_rep_state[blank_pos]
                    )
                    successor_beliefs.append((frozenset(new_belief), new_rep_state, (blank_pos, new_pos)))
        return successor_beliefs

    def heuristic(belief_state):
        return min(h_manhattan(list(state)) for state in belief_state)

    belief_state = frozenset(tuple(state) for state in valid_start_states)
    if not belief_state:
        print("Debug: Không có trạng thái ban đầu hợp lệ sau khi lọc.")
        return None, 0, time.time() - start_time
    visited.add(belief_state)
    representative_state = valid_start_states[0]

    queue = [(heuristic(belief_state), 0, belief_state, [], representative_state, None)]

    while queue:
        f_score, g_score, current_belief, path, rep_state, last_move = heapq.heappop(queue)

        if is_goal_belief(current_belief):
            print("Debug: Đã đạt trạng thái mục tiêu.")
            return path + [rep_state], len(visited), time.time() - start_time

        successor_beliefs = get_consistent_successors(current_belief)
        for next_belief, new_rep_state, move_positions in successor_beliefs:
            if next_belief not in visited:
                visited.add(next_belief)
                h_value = heuristic(next_belief)
                new_g_score = g_score + 1
                new_f_score = new_g_score + h_value
                heapq.heappush(
                    queue,
                    (new_f_score, new_g_score, next_belief, path + [rep_state], new_rep_state, move_positions)
                )
                print(f"Debug: Thêm trạng thái niềm tin mới, kích thước: {len(next_belief)}, f_score: {new_f_score}")

        if len(visited) > 20000:
            print("Debug: Đạt giới hạn số trạng thái thăm (20000).")
            break

    print("Debug: Không tìm thấy lời giải.")
    return None, len(visited), time.time() - start_time


def and_or_tree_search(start_state):
    # SỬA: Cải thiện để tìm giải pháp và giảm trạng thái thăm
    if not is_solvable(start_state):
        return None, 0, 0
    
    def search(state, path, visited, depth_limit, h_threshold):
        # Kiểm tra mục tiêu ngay đầu
        if is_goal(state):
            return path + [state], True
        
        if depth_limit <= 0:
            return None, False
        
        state_tuple = tuple(state)
        if state_tuple in visited:
            return None, False
        visited.add(state_tuple)
        
        # THÊM: Kiểm tra heuristic để cắt nhánh không triển vọng
        h = h_manhattan(state)
        if h > h_threshold:
            return None, False
        
        # Sắp xếp trạng thái láng giềng theo heuristic
        next_states = get_next_states(state)
        next_states.sort(key=lambda x: h_manhattan(x[0]))
        
        # THÊM: Giảm ngưỡng heuristic cho các nhánh sâu hơn
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


def backtracking(start_state, depth_limit=50):
    # SỬA: Cải thiện để tìm giải pháp và giảm trạng thái thăm
    if not is_solvable(start_state):
        return None, 0, 0
    
    def backtrack(state, path, visited, depth, h_threshold):
        # Kiểm tra mục tiêu ngay đầu
        if is_goal(state):
            return path + [state]
        
        if depth >= depth_limit:
            return None
        
        state_tuple = tuple(state)
        if state_tuple in visited:
            return None
        visited.add(state_tuple)
        
        # THÊM: Kiểm tra heuristic để cắt nhánh không triển vọng
        h = h_manhattan(state)
        if h > h_threshold:
            return None
        
        # THÊM: Sắp xếp trạng thái láng giềng theo heuristic
        next_states = get_next_states(state)
        next_states.sort(key=lambda x: h_manhattan(x[0]))
        
        # THÊM: Giảm ngưỡng heuristic cho các nhánh sâu hơn
        new_h_threshold = max(h_threshold - 1, 0) if h > 0 else h_threshold
        
        for next_state, _, _ in next_states:
            result = backtrack(next_state, path + [state], visited, depth + 1, new_h_threshold)
            if result:
                return result
        return None
    
    start_time = time.time()
    visited = set()
    initial_h = h_manhattan(start_state)
    h_threshold = initial_h + 10  # THÊM: Ngưỡng heuristic động
    
    # THÊM: Kiểm tra trạng thái ban đầu
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
            # Forward checking: Bỏ qua nếu trạng thái không khả nghiệm hoặc heuristic xấu đi
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
        
        # Lấy các trạng thái lân cận
        next_states = [(state, h_manhattan(state)) for state, _, _ in get_next_states(current_state) if tuple(state) not in visited]
        if not next_states:
            return None, len(visited), time.time() - start_time
        
        # Chọn trạng thái có ít xung đột nhất
        next_states.sort(key=lambda x: x[1])
        current_state = next_states[0][0]
        visited.add(tuple(current_state))
        path.append(current_state)
    
    return None, len(visited), time.time() - start_time
def simulated_annealing(start_state):
    # SỬA: Cải thiện thuật toán để tăng khả năng tìm trạng thái mục tiêu
    if not is_solvable(start_state):
        return None, 0, 0
    
    def acceptance_probability(old_cost, new_cost, T):
        if new_cost < old_cost:
            return 1.0
        return math.exp((old_cost - new_cost) / T)
    
    max_restarts = 5  # SỬA: Tăng từ 3 lên 5
    best_path = None
    best_visited_count = 0
    best_cost = float('inf')
    start_time = time.time()
    
    for _ in range(max_restarts):
        current_state = start_state.copy()
        visited = set()
        visited.add(tuple(current_state))
        path = [current_state]
        old_cost = h_manhattan(current_state)
        T = 20.0  # SỬA: Tăng từ 10.0 lên 20.0
        T_min = 0.00001
        alpha = 0.9  # SỬA: Giảm từ 0.95 xuống 0.9 để quay lại ví dụ gốc nhưng vẫn hiệu quả
        
        while T > T_min:
            i = 1
            while i <= 500:  # SỬA: Tăng từ 200 lên 500
                next_states = get_next_states(current_state)
                if not next_states:
                    break
                new_state, _, _ = random.choice(next_states)
                new_cost = h_manhattan(new_state)
                
                ap = acceptance_probability(old_cost, new_cost, T)
                if ap > random.random():
                    current_state = new_state
                    old_cost = new_cost
                    if tuple(current_state) not in visited:
                        visited.add(tuple(current_state))
                        path.append(current_state)
                
                if is_goal(current_state):
                    return path, len(visited), time.time() - start_time
                
                # THÊM: Theo dõi trạng thái tốt nhất
                if old_cost < best_cost:
                    best_cost = old_cost
                    best_path = path.copy()
                    best_visited_count = len(visited)
                
                i += 1
            T = T * alpha
        
        # SỬA: Cập nhật best_path nếu đường đi hiện tại tốt hơn
        if not best_path or (path and (is_goal(path[-1]) or h_manhattan(path[-1]) < best_cost)):
            best_cost = h_manhattan(path[-1])
            best_path = path
            best_visited_count = len(visited)
    
    return best_path, best_visited_count, time.time() - start_time

def genetic_algorithm(start_state, population_size=100, generations=1000, mutation_rate=0.1):
    if not is_solvable(start_state):
        print("Debug: Initial state is not solvable")
        return None, 0, 0
    
    start_time = time.time()
    visited = set()
    visited.add(tuple(start_state))

    def generate_population(size, initial_state):
        population = []
        for _ in range(size):
            state = initial_state.copy()
            path = [state]
            # Generate a random path of 5-15 steps
            for _ in range(random.randint(5, 15)):
                next_states = get_next_states(state)
                state, _, _ = random.choice(next_states)
                if tuple(state) not in visited:
                    visited.add(tuple(state))
                path.append(state)
            population.append((state, path))
        return population
    
    def fitness(state):
        # Improved fitness: Higher value for states closer to goal
        manhattan = h_manhattan(state)
        return 1 / (1 + manhattan) * 100  # Scale to make differences more significant
    
    def crossover(parent1, parent2):
        # Ensure parents have at least 2 states
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1  # Return parent1 if crossover is not possible
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child_path = parent1[:crossover_point] + parent2[crossover_point:]
        child_state = child_path[-1]
        # Only check solvability (sorted check is redundant)
        if not is_solvable(child_state):
            return parent1
        return child_path
    
    def mutate(path):
        if random.random() < mutation_rate:
            state = path[-1]
            next_states = get_next_states(state)
            new_state, _, _ = random.choice(next_states)
            path.append(new_state)
            if tuple(new_state) not in visited:
                visited.add(tuple(new_state))
        return path
    
    # Initialize population
    population = generate_population(population_size, start_state)
    
    for generation in range(generations):
        # Sort population by fitness
        population = sorted(population, key=lambda x: fitness(x[0]), reverse=True)
        
        # Check if the best individual is the goal
        best_state, best_path = population[0]
        if is_goal(best_state):
            print(f"Debug: Goal reached in generation {generation}")
            return best_path, len(visited), time.time() - start_time
        
        # Keep the top half of the population
        new_population = population[:population_size // 2]
        
        # Generate new individuals through crossover and mutation
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population[:population_size // 2], k=2)
            child_path = crossover(parent1[1], parent2[1])
            child_path = mutate(child_path)
            child_state = child_path[-1]
            new_population.append((child_state, child_path))
            if tuple(child_state) not in visited:
                visited.add(tuple(child_state))
        
        population = new_population
    
    # Return the best path found if no solution
    best_state, best_path = population[0]
    print("Debug: No solution found, returning best path")
    return best_path if best_path and is_solvable(best_path[-1]) else None, len(visited), time.time() - start_time
def partially_observable_search(start_state, observed_positions=None, max_states_visited=10000, max_time=30):
    if not is_solvable(start_state):
        print("Debug: Initial state is not solvable")
        return None, 0, 0

    start_time = time.time()
    visited = set()

    # Default observed positions
    if observed_positions is None:
        observed_positions = [0, 1, 3, 4]
    print(f"Debug: Observed positions: {observed_positions}")

    # Initialize belief state with the start state
    initial_belief_state = {tuple(start_state)}
    belief_tuple = frozenset(initial_belief_state)
    visited.add(belief_tuple)
    states_visited = 0

    # Priority queue: (heuristic, belief_state, path, representative_state)
    queue = [(h_manhattan(start_state), initial_belief_state, [], start_state)]

    # Define moves (up, down, left, right)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def get_observation(state):
        """Return observation: position of empty tile and adjacent tiles."""
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
        """Apply an action to all states in the belief state."""
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
        """Check if all states in belief state are the goal state."""
        for state_tuple in belief_state:
            if list(state_tuple) != GOAL_STATE:
                return False
        return True

    while queue and states_visited < max_states_visited and time.time() - start_time < max_time:
        _, current_belief, path, rep_state = heapq.heappop(queue)
        print(f"Debug: Belief size: {len(current_belief)}, Path length: {len(path)}")

        if is_goal_belief(current_belief):
            print("Debug: Goal state reached")
            return path + [rep_state], states_visited, time.time() - start_time

        for dx, dy in moves:
            new_belief_state = apply_action(current_belief, (dx, dy))
            if not new_belief_state:
                print(f"Debug: No new states for action ({dx}, {dy})")
                continue

            # Filter states based on observation
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
                    # Use average heuristic for the belief state
                    h_value = sum(h_manhattan(list(s)) for s in filtered_belief_state) / len(filtered_belief_state)
                    new_rep_state = list(next(iter(filtered_belief_state)))
                    heapq.heappush(queue, (h_value, filtered_belief_state, path + [rep_state], new_rep_state))
                    print(f"Debug: Added new belief state, size: {len(filtered_belief_state)}")

    print("Debug: No solution found")
    return None, states_visited, time.time() - start_time
def q_learning(start_state, max_episodes=1000, max_steps=100, alpha=0.1, gamma=0.9, epsilon=0.1):
    if not is_solvable(start_state):
        print("Debug: Initial state is not solvable")
        return None, 0, 0

    start_time = time.time()
    visited = set()
    q_table = {}  # Bảng Q: {state_tuple: {action: q_value}}

    def get_actions(state):
        """Lấy danh sách trạng thái láng giềng làm hành động."""
        return get_next_states(state)  # [(next_state, blank_pos, new_pos), ...]

    def choose_action(state, actions):
        """Chọn hành động theo epsilon-greedy."""
        state_tuple = tuple(state)
        if state_tuple not in q_table:
            q_table[state_tuple] = {i: 0.0 for i in range(len(actions))}
        if random.random() < epsilon:
            action_idx = random.randint(0, len(actions) - 1)
        else:
            q_values = q_table[state_tuple]
            action_idx = max(q_values, key=q_values.get)
        return action_idx, actions[action_idx]

    def get_reward(state, next_state):
        """Tính phần thưởng dựa trên trạng thái hiện tại và tiếp theo."""
        if is_goal(next_state):
            return 100
        current_h = h_manhattan(state)
        next_h = h_manhattan(next_state)
        return -1 + (current_h - next_h) * 2

    # Huấn luyện Q-Learning
    for episode in range(max_episodes):
        current_state = start_state.copy()
        state_tuple = tuple(current_state)
        if state_tuple not in visited:
            visited.add(state_tuple)

        for step in range(max_steps):
            actions = get_actions(current_state)
            if not actions:
                break
            action_idx, (next_state, _, _) = choose_action(current_state, actions)

            # Cập nhật visited
            next_state_tuple = tuple(next_state)
            if next_state_tuple not in visited:
                visited.add(next_state_tuple)

            # Tính phần thưởng
            reward = get_reward(current_state, next_state)

            # Cập nhật Q-value
            if next_state_tuple not in q_table:
                next_actions = get_actions(next_state)
                q_table[next_state_tuple] = {i: 0.0 for i in range(len(next_actions))}
            future_q = max(q_table[next_state_tuple].values()) if q_table[next_state_tuple] else 0
            current_q = q_table[state_tuple][action_idx]
            q_table[state_tuple][action_idx] = current_q + alpha * (reward + gamma * future_q - current_q)

            current_state = next_state
            state_tuple = next_state_tuple

            if is_goal(current_state):
                break

        # Kiểm tra giới hạn
        if len(visited) >= 10000 or time.time() - start_time >= 30:
            break

    # Suy ra đường đi tối ưu
    path = [start_state.copy()]
    current_state = start_state.copy()
    path_visited = set([tuple(current_state)])
    max_steps = 200

    while not is_goal(current_state) and max_steps > 0:
        actions = get_actions(current_state)
        if not actions:
            break
        state_tuple = tuple(current_state)
        if state_tuple not in q_table:
            break
        action_idx = max(q_table[state_tuple], key=q_table[state_tuple].get)
        next_state, _, _ = actions[action_idx]

        next_state_tuple = tuple(next_state)
        if next_state_tuple in path_visited:
            break  # Tránh vòng lặp
        path_visited.add(next_state_tuple)
        if next_state_tuple not in visited:
            visited.add(next_state_tuple)

        path.append(next_state)
        current_state = next_state
        max_steps -= 1

    runtime = time.time() - start_time
    if is_goal(current_state):
        print(f"Debug: Q-Learning found solution with {len(path)-1} steps")
        return path, len(visited), runtime
    print("Debug: Q-Learning failed to find solution")
    return None, len(visited), runtime

def sarsa(start_state, max_episodes=1000, max_steps=100, alpha=0.1, gamma=0.9, epsilon=0.1, max_states_visited=10000, max_time=30):
    if not is_solvable(start_state):
        print("Debug: Initial state is not solvable")
        return None, 0, 0

    start_time = time.time()
    visited = set()
    q_table = {}  # Bảng Q: {state_tuple: {action_idx: q_value}}

    # Ánh xạ di chuyển để làm rõ hành động
    move_names = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    print(f"SARSA: Starting training with initial state: {start_state}")

    def get_actions(state):
        """Lấy danh sách trạng thái láng giềng làm hành động, kèm chỉ số di chuyển."""
        next_states = get_next_states(state)  # [(next_state, blank_pos, new_pos), ...]
        actions = []
        blank_pos = find_blank(state)
        x, y = divmod(blank_pos, 3)
        for idx, (next_state, _, new_pos) in enumerate(next_states):
            new_x, new_y = divmod(new_pos, 3)
            for move_idx, (dx, dy) in enumerate(moves):
                if new_x == x + dx and new_y == y + dy:
                    actions.append((move_idx, next_state))
                    break
        return actions  # [(move_idx, next_state), ...]

    def choose_action(state, actions):
        """Chọn hành động theo epsilon-greedy."""
        state_tuple = tuple(state)
        if state_tuple not in q_table:
            q_table[state_tuple] = {move_idx: 0.0 for move_idx, _ in actions}
            print(f"SARSA: Initialized Q-values for state {state_tuple}: {q_table[state_tuple]}")
        if random.random() < epsilon:
            move_idx, next_state = random.choice(actions)
            print(f"SARSA: Chose random action ({move_names[move_idx]})")
        else:
            q_values = q_table[state_tuple]
            move_idx = max(q_values, key=q_values.get)
            next_state = next(s for m, s in actions if m == move_idx)
            print(f"SARSA: Chose optimal action ({move_names[move_idx]}), Q-values: {q_values}")
        return move_idx, next_state

    def get_reward(state, next_state):
        """Tính phần thưởng dựa trên trạng thái hiện tại và tiếp theo."""
        if is_goal(next_state):
            print("SARSA: Reached goal state, reward: 100")
            return 100
        current_h = h_manhattan(state)
        next_h = h_manhattan(next_state)
        reward = -1 if next_h >= current_h else (current_h - next_h) * 2
        print(f"SARSA: Reward: {reward}, Current heuristic: {current_h}, Next heuristic: {next_h}")
        return reward

    # Huấn luyện SARSA
    for episode in range(max_episodes):
        if len(visited) >= max_states_visited or time.time() - start_time >= max_time:
            print(f"SARSA: Stopped training after {episode} episodes, "
                  f"States visited: {len(visited)}, Time: {time.time() - start_time:.2f}s")
            break

        print(f"SARSA: Starting episode {episode + 1}")
        current_state = start_state.copy()
        state_tuple = tuple(current_state)
        if state_tuple not in visited:
            visited.add(state_tuple)

        # Chọn hành động đầu tiên
        actions = get_actions(current_state)
        if not actions:
            print("SARSA: No valid actions, ending episode")
            break
        move_idx, next_state = choose_action(current_state, actions)

        for step in range(max_steps):
            # Thực hiện hành động, nhận phần thưởng và trạng thái tiếp theo
            reward = get_reward(current_state, next_state)
            next_state_tuple = tuple(next_state)
            if next_state_tuple not in visited:
                visited.add(next_state_tuple)

            # Chọn hành động tiếp theo cho trạng thái tiếp theo
            next_actions = get_actions(next_state)
            if not next_actions:
                print("SARSA: No valid actions for next state, ending episode")
                break
            next_move_idx, next_next_state = choose_action(next_state, next_actions)

            # Cập nhật Q-value
            if next_state_tuple not in q_table:
                q_table[next_state_tuple] = {m: 0.0 for m, _ in next_actions}
                print(f"SARSA: Initialized Q-values for new state {next_state_tuple}")
            next_q = q_table[next_state_tuple][next_move_idx]
            current_q = q_table[state_tuple][move_idx]
            q_table[state_tuple][move_idx] = current_q + alpha * (reward + gamma * next_q - current_q)
            print(f"SARSA: Updated Q-value for state {state_tuple}, action {move_names[move_idx]}: {q_table[state_tuple][move_idx]}")

            # Chuyển sang trạng thái và hành động tiếp theo
            current_state = next_state
            state_tuple = next_state_tuple
            move_idx = next_move_idx
            next_state = next_next_state

            if is_goal(current_state):
                print(f"SARSA: Reached goal in episode {episode + 1}, step {step + 1}")
                break

    # Suy ra đường đi tối ưu
    print("SARSA: Starting to infer optimal path")
    path = [start_state.copy()]
    current_state = start_state.copy()
    path_visited = set([tuple(current_state)])
    max_steps = 200

    while not is_goal(current_state) and max_steps > 0:
        if len(visited) >= max_states_visited or time.time() - start_time >= max_time:
            print(f"SARSA: Stopped path inference, States visited: {len(visited)}, Time: {time.time() - start_time:.2f}s")
            return None, len(visited), time.time() - start_time

        actions = get_actions(current_state)
        if not actions:
            print("SARSA: No valid actions in path inference")
            break
        state_tuple = tuple(current_state)
        if state_tuple not in q_table:
            print(f"SARSA: State {state_tuple} not in q_table, stopping inference")
            break
        q_values = q_table[state_tuple]
        move_idx = max(q_values, key=q_values.get)
        try:
            next_state = next(s for m, s in actions if m == move_idx)
        except StopIteration:
            print(f"SARSA: No matching next state for move_idx {move_idx}, stopping inference")
            break

        next_state_tuple = tuple(next_state)
        if next_state_tuple in path_visited:
            print(f"SARSA: Detected loop at state {next_state_tuple}, stopping inference")
            break
        path_visited.add(next_state_tuple)
        if next_state_tuple not in visited:
            visited.add(next_state_tuple)

        path.append(next_state)
        current_state = next_state
        print(f"SARSA: Added step to path: {move_names[move_idx]}, New state: {current_state}")
        max_steps -= 1

    runtime = time.time() - start_time
    if is_goal(current_state):
        print(f"SARSA: Found solution with {len(path)-1} steps, States visited: {len(visited)}, Time: {runtime:.2f}s")
        return path, len(visited), runtime
    print(f"SARSA: Failed to find solution, States visited: {len(visited)}, Time: {runtime:.2f}s")
    return None, len(visited), runtime

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
        default_state = [2, 6, 5, 8, 0, 7, 4, 3, 1]
        for i in range(3):
            for j in range(3):
                self.entries[i][j].insert(0, str(default_state[i * 3 + j]))

        # Algorithm selection frame
        self.algorithm_frame = ttk.LabelFrame(self.main_frame, text="SELECT ALGORITHMS", padding=10)
        self.algorithm_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.algorithm_vars = {}
        algorithms = ["BFS", "DFS","UCS", "Greedy", "A*", "IDA*", "IDS", "Simple HC", "Steepest HC", "Stochastic HC", "Belief State", "AND-OR Tree", "Backtracking", "Forward Checking", "Min-Conflicts", "Simulated Annealing", "Genetic","Beam Search","Partially Observable","Q-Learning","SARSA"]
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
        
        for algorithm in selected_algorithms:
            print(f"\n=== {algorithm} ===")
            if algorithm == "BFS":
                solution, visited_count, runtime = bfs(state)
            elif algorithm == "DFS":
                solution, visited_count, runtime = dfs(state)
            elif algorithm == "Greedy":
                solution, visited_count, runtime = greedy(state)
            elif algorithm == "A*":
                solution, visited_count, runtime = a_star(state)
            elif algorithm == "IDA*":
                solution, visited_count, runtime = ida_star(state)
            elif algorithm == "IDS":
                solution, visited_count, runtime = ids(state)
            elif algorithm == "Simple HC":
                solution, visited_count, runtime = simple_hill_climbing(state)
            elif algorithm == "Steepest HC":
                solution, visited_count, runtime = steepest_ascent_hill_climbing(state)
            elif algorithm == "Stochastic HC":
                solution, visited_count, runtime = stochastic_hill_climbing(state)
            elif algorithm == "AND-OR Tree":
                solution, visited_count, runtime = and_or_tree_search(state)
            elif algorithm == "Backtracking":
                solution, visited_count, runtime = backtracking(state)
            elif algorithm == "Forward Checking":
                solution, visited_count, runtime = forward_checking(state)
            elif algorithm == "Min-Conflicts":
                solution, visited_count, runtime = min_conflicts(state)
            elif algorithm == "Simulated Annealing":
                solution, visited_count, runtime = simulated_annealing(state)
            elif algorithm == "Genetic":
                solution, visited_count, runtime = genetic_algorithm(state)
            elif algorithm == "Beam Search":
                solution, visited_count, runtime = beam_search(state)
            elif algorithm == "UCS":
                solution, visited_count, runtime = uniform_cost_search(state)
            elif algorithm == "Partially Observable":
                solution, visited_count, runtime = partially_observable_search(state, observed_positions=[0, 1, 3, 4])
            elif algorithm == "Q-Learning":
                solution, visited_count, runtime = q_learning(state)
            elif algorithm == "SARSA":
                solution, visited_count, runtime = sarsa(state)
            
            
            
            
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
            messagebox.showinfo("Kết quả", "Không tìm thấy giải pháp cho các thuật toán đã chọn!")
        else:
            self.auto_play()

    def auto_play(self):
        selected_algorithm = next((algo for algo, (solution, _, _) in self.solutions.items() if solution), None)
        if not selected_algorithm:
            return
        
        self.is_auto_playing = True
        solution = self.solutions[selected_algorithm][0]
        delay = int(self.speed_var.get() * 1000)  # Chuyển sang mili giây

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