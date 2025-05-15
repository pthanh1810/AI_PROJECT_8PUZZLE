import pygame
import sys
from pygame.locals import *
import copy
import time
import random
import tkinter as tk
from tkinter import ttk
import math
from collections import deque

# Khởi tạo Pygame và kiểm tra
pygame.init()
if not pygame.get_init():
    print("Pygame initialization failed!")
    sys.exit(1)

# Cài đặt màn hình
WIDTH = 1080
HEIGHT = 600
CELL_SIZE = 120
GRID_SIZE = 3
BUTTON_WIDTH = 250
BUTTON_HEIGHT = 50
CONTROL_PANEL_WIDTH = 600
CONTROL_PANEL_X = 470
screen = pygame.display.set_mode((WIDTH + 30, HEIGHT))
pygame.display.set_caption("8-Puzzle Solver of Lê Vũ Hải")

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
BLUE = (70, 130, 180)
LIGHT_BLUE = (173, 216, 230)
GREEN = (60, 179, 113)
RED = (220, 20, 60)
YELLOW = (255, 215, 0)
PURPLE = (147, 112, 219)
ORANGE = (255, 165, 0)
DARK_PURPLE = (106, 90, 205)
LIGHT_PURPLE = (186, 147, 255)
DARK_GREEN = (34, 139, 34)
LIGHT_GREEN = (144, 238, 144)
DARK_YELLOW = (218, 165, 32)
LIGHT_YELLOW = (255, 255, 102)
DARK_RED = (178, 34, 34)
LIGHT_RED = (255, 99, 71)
NEXT_MOVE_COLOR = (255, 182, 193)

# Font hỗ trợ tiếng Việt (Arial)
try:
    FONT_PATH = "C:\\Windows\\Fonts\\arial.ttf"
    FONT = pygame.font.Font(FONT_PATH, 36)
    TITLE_FONT = pygame.font.Font(FONT_PATH, 48)
    NUMBER_FONT = pygame.font.Font(FONT_PATH, 60)
    SMALL_FONT = pygame.font.Font(FONT_PATH, 24)
except:
    print("Không tìm thấy font Arial. Sử dụng font mặc định.")
    FONT = pygame.font.Font(None, 36)
    TITLE_FONT = pygame.font.Font(None, 48)
    NUMBER_FONT = pygame.font.Font(None, 60)
    SMALL_FONT = pygame.font.Font(None, 24)

# Tải hình nền
picture_background = None
try:
    picture_background = pygame.image.load(r"download.png")
    picture_background = pygame.transform.scale(picture_background, (150, 150))
except:
    print("Không tìm thấy hình nền! Sử dụng màu trắng làm nền.")

# Trạng thái ban đầu và mục tiêu
INITIAL_STATE = [
    [1, 2, 3],
    [4, 0, 8],
    [5, 6, 7]
]
GOAL_STATE = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

# Danh sách thuật toán Local Search
ALGORITHMS = [
    "Simple Hill Climbing",
    "Stochastic Hill Climbing",
    "Steepest Hill Climbing",
    "Simulated Annealing",
    "Local Beam Search"
]

def manhattan_distance(state, goal):
    distance = 0
    for i in range(3):
        for j in range(3):
            value = state[i][j]
            if value != 0:
                goal_pos = [(x, y) for x in range(3) for y in range(3) if goal[x][y] == value][0]
                distance += abs(i - goal_pos[0]) + abs(j - goal_pos[1])
    return distance

def get_neighbors(state):
    neighbors = []
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 3 and 0 <= nj < 3:
                        new_state = [row[:] for row in state]
                        new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
                        neighbors.append(new_state)
                return neighbors
    return []

class DropdownMenu:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Chọn Thuật Toán")
        self.root.geometry("300x150")
        self.root.configure(bg="#f0f0f0")

        label = tk.Label(self.root, text="Chọn một thuật toán", font=("Arial", 12), bg="#f0f0f0")
        label.pack(pady=10)

        self.selected_algorithm = tk.StringVar()
        self.combobox = ttk.Combobox(
            self.root,
            textvariable=self.selected_algorithm,
            values=ALGORITHMS,
            state="readonly",
            font=("Arial", 10),
            width=30
        )
        self.combobox.set("Simple Hill Climbing")
        self.combobox.pack(pady=10)

        style = ttk.Style()
        style.configure("TButton", font=("Arial", 10))
        self.button = ttk.Button(self.root, text="Xác nhận", command=self.root.quit)
        self.button.pack(pady=10)

    def get_selection(self):
        self.root.mainloop()
        return self.selected_algorithm.get()

class Puzzle:
    def __init__(self, initial_state):
        self.state = initial_state
        self.move_count = 0
        self.execution_time = 0

    def draw(self, screen, highlight_pos=None):
        for i in range(3):
            for j in range(3):
                x0 = j * CELL_SIZE + 50
                y0 = i * CELL_SIZE + 50
                rect = pygame.Rect(x0, y0, CELL_SIZE, CELL_SIZE)
                if self.state[i][j] == 0:
                    pygame.draw.rect(screen, GRAY, rect, border_radius=10)
                elif highlight_pos and (i, j) == highlight_pos:
                    pygame.draw.rect(screen, NEXT_MOVE_COLOR, rect, border_radius=10)
                else:
                    pygame.draw.rect(screen, LIGHT_BLUE, rect, border_radius=10)
                pygame.draw.rect(screen, DARK_GRAY, rect, 3, border_radius=10)
                if self.state[i][j] != 0:
                    text = NUMBER_FONT.render(str(self.state[i][j]), True, WHITE)
                    text_rect = text.get_rect(center=(x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2))
                    screen.blit(text, text_rect)

    def draw_best_individual(self, screen, state, x_offset, y_offset):
        small_cell_size = CELL_SIZE // 2
        for i in range(3):
            for j in range(3):
                x0 = x_offset + j * small_cell_size
                y0 = y_offset + i * small_cell_size
                rect = pygame.Rect(x0, y0, small_cell_size, small_cell_size)
                if state[i][j] == 0:
                    pygame.draw.rect(screen, GRAY, rect, border_radius=5)
                else:
                    pygame.draw.rect(screen, LIGHT_BLUE, rect, border_radius=5)
                pygame.draw.rect(screen, DARK_GRAY, rect, 2, border_radius=5)
                if state[i][j] != 0:
                    text = SMALL_FONT.render(str(state[i][j]), True, WHITE)
                    text_rect = text.get_rect(center=(x0 + small_cell_size // 2, y0 + small_cell_size // 2))
                    screen.blit(text, text_rect)

    def find_empty(self):
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == 0:
                    return i, j
        return None

    def get_state_tuple(self):
        return tuple(tuple(row) for row in self.state)

class PuzzleSolver:
    def __init__(self, algorithm, puzzle):
        self.algorithm = algorithm
        self.puzzle = puzzle
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def hill_climbing_simple(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []
        current_state = [row[:] for row in self.puzzle.state]
        path = [current_state]
        max_iterations = 100
        for _ in range(max_iterations):
            if current_state == GOAL_STATE:
                return path, []
            neighbors = get_neighbors(current_state)
            current_h = manhattan_distance(current_state, GOAL_STATE)
            for neighbor in neighbors:
                neighbor_h = manhattan_distance(neighbor, GOAL_STATE)
                if neighbor_h < current_h:
                    current_state = neighbor
                    path.append(neighbor)
                    self.puzzle.state = current_state
                    if picture_background:
                        screen.fill(WHITE)
                        screen.blit(picture_background, (150, 450))
                    else:
                        screen.fill(WHITE)
                    self.puzzle.draw(screen)
                    draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, [], 0)
                    pygame.display.flip()
                    pygame.time.wait(1000)
                    break
            else:
                return path, []
        return path, []

    def hill_climbing_stochastic(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []
        current_state = [row[:] for row in self.puzzle.state]
        path = [current_state]
        max_iterations = 100
        for _ in range(max_iterations):
            if current_state == GOAL_STATE:
                return path, []
            neighbors = get_neighbors(current_state)
            current_h = manhattan_distance(current_state, GOAL_STATE)
            better_neighbors = [(n, manhattan_distance(n, GOAL_STATE))
                                for n in neighbors if manhattan_distance(n, GOAL_STATE) < current_h]
            if better_neighbors:
                next_state = random.choice([n for n, _ in better_neighbors])
                current_state = next_state
                path.append(next_state)
                self.puzzle.state = current_state
                if picture_background:
                    screen.fill(WHITE)
                    screen.blit(picture_background, (150, 450))
                else:
                    screen.fill(WHITE)
                self.puzzle.draw(screen)
                draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, [], 0)
                pygame.display.flip()
                pygame.time.wait(1000)
            else:
                return path, []
        return path, []

    def hill_climbing_steepest(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []
        current_state = [row[:] for row in self.puzzle.state]
        path = [current_state]
        max_iterations = 100
        for _ in range(max_iterations):
            if current_state == GOAL_STATE:
                return path, []
            neighbors = get_neighbors(current_state)
            current_h = manhattan_distance(current_state, GOAL_STATE)
            best_neighbor = min(neighbors,
                                key=lambda x: manhattan_distance(x, GOAL_STATE),
                                default=None)
            best_h = manhattan_distance(best_neighbor, GOAL_STATE) if best_neighbor else float('inf')
            if best_h < current_h:
                current_state = best_neighbor
                path.append(best_neighbor)
                self.puzzle.state = current_state
                if picture_background:
                    screen.fill(WHITE)
                    screen.blit(picture_background, (150, 450))
                else:
                    screen.fill(WHITE)
                self.puzzle.draw(screen)
                draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, [], 0)
                pygame.display.flip()
                pygame.time.wait(1000)
            else:
                return path, []
        return path, []

    def simulated_annealing(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []
        initial_temp = 1000.0
        alpha = 0.995
        max_iterations = 10000
        current_state = [row[:] for row in self.puzzle.state]
        best_state = current_state
        best_cost = manhattan_distance(current_state, GOAL_STATE)
        path = [current_state]
        best_individuals = [(current_state, 0, -best_cost)]
        temp = initial_temp
        iteration = 0
        while iteration < max_iterations and temp > 0.1:
            neighbors = get_neighbors(current_state)
            next_state = random.choice(neighbors)
            current_cost = manhattan_distance(current_state, GOAL_STATE)
            next_cost = manhattan_distance(next_state, GOAL_STATE)
            delta_e = next_cost - current_cost
            if next_state == GOAL_STATE:
                path.append(next_state)
                best_individuals.append((next_state, iteration, 0))
                print(f"Simulated Annealing found solution after {iteration} iterations")
                self.puzzle.state = next_state
                if picture_background:
                    screen.fill(WHITE)
                    screen.blit(picture_background, (150, 450))
                else:
                    screen.fill(WHITE)
                self.puzzle.draw(screen)
                draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, best_individuals,
                        len(best_individuals) - 1)
                pygame.display.flip()
                pygame.time.wait(5000)
                return path, best_individuals
            if delta_e <= 0 or random.random() < math.exp(-delta_e / temp):
                current_state = [row[:] for row in next_state]
                path.append(current_state)
                if next_cost < best_cost:
                    best_state = [row[:] for row in next_state]
                    best_cost = next_cost
                    best_individuals.append((best_state, iteration, -best_cost))
                    self.puzzle.state = best_state
                    if picture_background:
                        screen.fill(WHITE)
                        screen.blit(picture_background, (150, 450))
                    else:
                        screen.fill(WHITE)
                    self.puzzle.draw(screen)
                    draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, best_individuals,
                            len(best_individuals) - 1)
                    pygame.display.flip()
                    pygame.time.wait(1000)
            temp *= alpha
            iteration += 1
        print("Simulated Annealing không tìm thấy giải pháp chính xác, trả về trạng thái tốt nhất.")
        path.append(best_state)
        best_individuals.append((best_state, iteration - 1, -best_cost))
        self.puzzle.state = best_state
        if picture_background:
            screen.fill(WHITE)
            screen.blit(picture_background, (150, 450))
        else:
            screen.fill(WHITE)
        self.puzzle.draw(screen)
        draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, best_individuals,
                len(best_individuals) - 1)
        pygame.display.flip()
        pygame.time.wait(5000)
        return path, best_individuals

    def local_beam_search(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []
        def generate_individual():
            state = [row[:] for row in self.puzzle.state]
            for _ in range(random.randint(5, 20)):
                neighbors = get_neighbors(state)
                state = random.choice(neighbors)
            return state
        def fitness(state):
            return -manhattan_distance(state, GOAL_STATE)
        k = 10
        max_iterations = 1000
        states = [generate_individual() for _ in range(k)]
        path = [self.puzzle.state]
        best_individuals = []
        best_state = states[0]
        best_fitness = fitness(best_state)
        best_individuals.append((best_state, 0, best_fitness))
        iteration = 0
        while iteration < max_iterations:
            successors = []
            for state in states:
                neighbors = get_neighbors(state)
                successors.extend(neighbors)
            for successor in successors:
                if successor == GOAL_STATE:
                    path.append(successor)
                    best_individuals.append((successor, iteration, 0))
                    print(f"Local Beam Search found solution after {iteration} iterations")
                    self.puzzle.state = successor
                    if picture_background:
                        screen.fill(WHITE)
                        screen.blit(picture_background, (150, 450))
                    else:
                        screen.fill(WHITE)
                    self.puzzle.draw(screen)
                    draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, best_individuals,
                            len(best_individuals) - 1)
                    pygame.display.flip()
                    pygame.time.wait(5000)
                    return path, best_individuals
            successor_tuples = [tuple(tuple(row) for row in s) for s in successors]
            unique_successors = []
            seen = set()
            for s, t in zip(successors, successor_tuples):
                if t not in seen:
                    unique_successors.append(s)
                    seen.add(t)
            if not unique_successors:
                print("No more unique successors available.")
                break
            successor_fitness = [(s, fitness(s)) for s in unique_successors]
            fitness_scores = [f for _, f in successor_fitness]
            min_fitness = min(fitness_scores)
            normalized_fitness = [f - min_fitness + 1 for f in fitness_scores]
            total_fitness = sum(normalized_fitness)
            probabilities = [f / total_fitness for f in normalized_fitness] if total_fitness > 0 else [1/len(normalized_fitness)] * len(normalized_fitness)
            new_states = []
            sorted_successors = sorted(successor_fitness, key=lambda x: x[1], reverse=True)
            top_k = min(k // 2, len(sorted_successors))
            for i in range(top_k):
                new_states.append(sorted_successors[i][0])
            remaining = k - len(new_states)
            if remaining > 0 and unique_successors:
                chosen = random.choices(
                    unique_successors,
                    weights=probabilities,
                    k=remaining
                )
                new_states.extend([s for s in chosen if s not in new_states])
            while len(new_states) < k and unique_successors:
                new_states.append(random.choice(unique_successors))
            states = new_states[:k]
            current_best = max(states, key=fitness, default=states[0])
            current_fitness = fitness(current_best)
            if current_fitness > best_fitness:
                best_state = [row[:] for row in current_best]
                best_fitness = current_fitness
                best_individuals.append((best_state, iteration, best_fitness))
                self.puzzle.state = best_state
                if picture_background:
                    screen.fill(WHITE)
                    screen.blit(picture_background, (150, 450))
                else:
                    screen.fill(WHITE)
                self.puzzle.draw(screen)
                draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, best_individuals,
                        len(best_individuals) - 1)
                pygame.display.flip()
                pygame.time.wait(1000)
            path.append(states[0])
            iteration += 1
        print("Local Beam Search không tìm thấy giải pháp chính xác, trả về trạng thái tốt nhất.")
        path.append(best_state)
        best_individuals.append((best_state, iteration - 1, best_fitness))
        self.puzzle.state = best_state
        if picture_background:
            screen.fill(WHITE)
            screen.blit(picture_background, (150, 450))
        else:
            screen.fill(WHITE)
        self.puzzle.draw(screen)
        draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, best_individuals,
                len(best_individuals) - 1)
        pygame.display.flip()
        pygame.time.wait(5000)
        return path, best_individuals

    def solve(self, screen, font, small_font, selected_algorithm):
        start_time = time.time()
        print(f"Starting {self.algorithm}...")
        path = None
        best_individuals = []
        if self.algorithm == "Simple Hill Climbing":
            path, best_individuals = self.hill_climbing_simple(screen, font, small_font, selected_algorithm)
        elif self.algorithm == "Stochastic Hill Climbing":
            path, best_individuals = self.hill_climbing_stochastic(screen, font, small_font, selected_algorithm)
        elif self.algorithm == "Steepest Hill Climbing":
            path, best_individuals = self.hill_climbing_steepest(screen, font, small_font, selected_algorithm)
        elif self.algorithm == "Simulated Annealing":
            path, best_individuals = self.simulated_annealing(screen, font, small_font, selected_algorithm)
        elif self.algorithm == "Local Beam Search":
            path, best_individuals = self.local_beam_search(screen, font, small_font, selected_algorithm)
        else:
            print(f"Thuật toán {self.algorithm} chưa được triển khai.")
            path = []
            best_individuals = []
        end_time = time.time()
        self.puzzle.execution_time = end_time - start_time
        if path:
            print(f"{self.algorithm} completed in {self.puzzle.execution_time:.2f}s")
        return path, best_individuals

def draw_gradient_rect(screen, rect, color1, color2):
    x, y, w, h = rect
    gradient_surface = pygame.Surface((w, h))
    for i in range(h):
        ratio = i / h
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        pygame.draw.line(gradient_surface, (r, g, b), (0, i), (w, i))
    screen.blit(gradient_surface, (x, y))

def draw_ui(screen, font, small_font, selected_algorithm, puzzle, is_running, is_paused, best_individuals, current_step):
    title_text = TITLE_FONT.render("Thông tin thuật toán", True, BLUE)
    screen.blit(title_text, (450, 20))
    shadow_rect = pygame.Rect(CONTROL_PANEL_X + 5, 85, CONTROL_PANEL_WIDTH, 480)
    pygame.draw.rect(screen, DARK_GRAY, shadow_rect, border_radius=15)
    pygame.draw.rect(screen, LIGHT_BLUE, (CONTROL_PANEL_X, 80, CONTROL_PANEL_WIDTH, 480), border_radius=15)
    pygame.draw.rect(screen, DARK_GRAY, (CONTROL_PANEL_X, 80, CONTROL_PANEL_WIDTH, 480), 3, border_radius=15)

    algo_text = font.render(f"Thuật toán: {selected_algorithm}", True, BLACK)
    algo_rect = algo_text.get_rect(topleft=(CONTROL_PANEL_X + 20, 100))
    screen.blit(algo_text, algo_rect)

    steps_text = font.render(f"Số bước thực hiện: {puzzle.move_count}", True, BLACK)
    steps_rect = steps_text.get_rect(topleft=(CONTROL_PANEL_X + 20, 150))
    screen.blit(steps_text, steps_rect)

    time_text = font.render(f"Thời gian: {puzzle.execution_time:.2f}s", True, BLACK)
    time_rect = time_text.get_rect(topleft=(CONTROL_PANEL_X + 20, 200))
    screen.blit(time_text, time_rect)

    button_x = CONTROL_PANEL_X + (CONTROL_PANEL_WIDTH - BUTTON_WIDTH) // 2
    select_rect = pygame.Rect(button_x, 290, BUTTON_WIDTH, BUTTON_HEIGHT)
    draw_gradient_rect(screen, select_rect, DARK_PURPLE, LIGHT_PURPLE)
    pygame.draw.rect(screen, DARK_GRAY, select_rect, 3, border_radius=10)
    select_text = font.render("Thuật toán", True, WHITE)
    select_text_rect = select_text.get_rect(center=select_rect.center)
    screen.blit(select_text, select_text_rect)

    start_rect = pygame.Rect(button_x, 360, BUTTON_WIDTH, BUTTON_HEIGHT)
    if not is_running:
        draw_gradient_rect(screen, start_rect, DARK_GREEN, LIGHT_GREEN)
    else:
        pygame.draw.rect(screen, GRAY, start_rect, border_radius=10)
    pygame.draw.rect(screen, DARK_GRAY, start_rect, 3, border_radius=10)
    start_text = font.render("Bắt đầu", True, WHITE)
    start_text_rect = start_text.get_rect(center=start_rect.center)
    screen.blit(start_text, start_text_rect)

    pause_rect = pygame.Rect(button_x, 430, BUTTON_WIDTH, BUTTON_HEIGHT)
    if not is_paused:
        draw_gradient_rect(screen, pause_rect, DARK_YELLOW, LIGHT_YELLOW)
    else:
        pygame.draw.rect(screen, ORANGE, pause_rect, border_radius=10)
    pygame.draw.rect(screen, DARK_GRAY, pause_rect, 3, border_radius=10)
    pause_text = font.render("Tạm Dừng" if not is_paused else "Tiếp Tục", True, WHITE)
    pause_text_rect = pause_text.get_rect(center=pause_rect.center)
    screen.blit(pause_text, pause_text_rect)

    reset_rect = pygame.Rect(button_x, 500, BUTTON_WIDTH, BUTTON_HEIGHT)
    draw_gradient_rect(screen, reset_rect, DARK_RED, LIGHT_RED)
    pygame.draw.rect(screen, DARK_GRAY, reset_rect, 3, border_radius=10)
    reset_text = font.render("Đặt Lại", True, WHITE)
    reset_text_rect = reset_text.get_rect(center=reset_rect.center)
    screen.blit(reset_text, reset_text_rect)

    if (selected_algorithm in ["Simulated Annealing", "Local Beam Search"]) and best_individuals and current_step < len(best_individuals):
        best_state, step_or_gen, best_fitness = best_individuals[current_step]
        label = "Lần lặp"
        best_text = small_font.render(f"Trạng thái tốt nhất ({label} {step_or_gen}, Fitness: {best_fitness})", True, BLACK)
        best_rect = best_text.get_rect(topleft=(CONTROL_PANEL_X + 20, 250))
        screen.blit(best_text, best_rect)
        puzzle.draw_best_individual(screen, best_state, CONTROL_PANEL_X + 20, 280)

    return select_rect, start_rect, pause_rect, reset_rect

def find_highlight_pos(puzzle, path, step):
    if step + 1 >= len(path):
        return None, None
    current = path[step]
    next_state = path[step + 1]
    empty_pos = None
    next_pos = None
    for i in range(3):
        for j in range(3):
            if current[i][j] == 0:
                empty_pos = (i, j)
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 3 and 0 <= nj < 3 and next_state[i][j] == current[ni][nj]:
                        next_pos = (ni, nj)
                        return empty_pos, next_pos
    return empty_pos, next_pos

def print_state(state, step):
    print(f"\nBước {step}:")
    for row in state:
        print(row)

def main():
    print("Starting main loop...")
    puzzle = Puzzle([row[:] for row in INITIAL_STATE])
    selected_algorithm = "Simple Hill Climbing"
    running = True
    solving = False
    paused = False
    path = []
    best_individuals = []
    step = 0
    clock = pygame.time.Clock()
    try:
        while running:
            if picture_background:
                screen.fill(WHITE)
                screen.blit(picture_background, (150, 450))
            else:
                screen.fill(WHITE)
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == MOUSEBUTTONDOWN:
                    x, y = event.pos
                    select_rect, start_rect, pause_rect, reset_rect = draw_ui(
                        screen, FONT, SMALL_FONT, selected_algorithm, puzzle, solving, paused, best_individuals, step
                    )
                    if select_rect.collidepoint(x, y) and not solving:
                        dropdown = DropdownMenu()
                        selected_algorithm = dropdown.get_selection()
                        dropdown.root.destroy()
                        puzzle = Puzzle([row[:] for row in INITIAL_STATE])
                    if start_rect.collidepoint(x, y) and not solving:
                        solving = True
                        paused = False
                        solver = PuzzleSolver(selected_algorithm, puzzle)
                        path, best_individuals = solver.solve(screen, FONT, SMALL_FONT, selected_algorithm)
                        step = 0
                        puzzle.move_count = 0
                        if not path:
                            print("Không tìm thấy giải pháp!")
                            solving = False
                        else:
                            print("Đã tìm thấy giải pháp! Bắt đầu in các trạng thái:")
                    if pause_rect.collidepoint(x, y) and solving:
                        paused = not paused
                    if reset_rect.collidepoint(x, y):
                        puzzle = Puzzle([row[:] for row in INITIAL_STATE])
                        puzzle.move_count = 0
                        puzzle.execution_time = 0
                        solving = False
                        paused = False
                        path = []
                        best_individuals = []
                        step = 0
                        print("Đã đặt lại trạng thái ban đầu:")
            if solving and not paused and path and step < len(path):
                empty_pos, next_pos = find_highlight_pos(puzzle, path, step)
                puzzle.state = [row[:] for row in path[step]]
                puzzle.move_count = step
                print_state(puzzle.state, step)
                step += 1
                pygame.time.wait(1000)
                if step >= len(path):
                    solving = False
                    print("Đã hoàn thành giải pháp!")
            empty_pos, next_pos = find_highlight_pos(puzzle, path, step - 1) if path and step > 0 else (None, None)
            puzzle.draw(screen, next_pos if next_pos and solving else None)
            draw_ui(screen, FONT, SMALL_FONT, selected_algorithm, puzzle, solving, paused, best_individuals, step - 1 if step > 0 else 0)
            pygame.display.flip()
            clock.tick(60)
    except KeyboardInterrupt:
        print("Chương trình bị dừng bởi người dùng (Ctrl+C)")
    finally:
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()