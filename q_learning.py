import pygame
import sys
import time
import random
from collections import defaultdict
from typing import Tuple, Dict, Set, List

State = Tuple[int, ...]
Action = str
GOAL: State = (1, 2, 3, 4, 5, 6, 7, 8, 0)  


def ke(x: State) -> List[Tuple[Action, State]]:
    neighbors = []
    z = x.index(0)
    r, c = divmod(z, 3)
    moves = [(1, 0, 'D'), (-1, 0, 'U'), (0, 1, 'R'), (0, -1, 'L')]
    for dr, dc, move in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            y = list(x)
            nz = nr * 3 + nc
            y[z], y[nz] = y[nz], y[z]
            neighbors.append((move, tuple(y)))
    return neighbors

def q_learning(start_node: State, goal_node: State = GOAL, episodes: int = 10000) -> Tuple[str, float]:
    alpha, gamma = 0.1, 0.9
    epsilon, epsilon_end, decay = 1.0, 0.01, 0.999
    max_steps = 200
    q_table: Dict[State, Dict[Action, float]] = defaultdict(lambda: defaultdict(float))
    start_time = time.perf_counter()

    for episode in range(episodes):
        current_state = start_node
        for _ in range(max_steps):
            if current_state == goal_node:
                break
            valid_moves = ke(current_state)
            if random.random() < epsilon:
                action, next_state = random.choice(valid_moves)
            else:
                q_vals = [(move, q_table[current_state][move]) for move, _ in valid_moves]
                max_q = max([q for _, q in q_vals], default=0)
                best = [mv for mv, q in q_vals if q == max_q]
                action = random.choice(best) if best else valid_moves[0][0]
                next_state = dict(valid_moves)[action]
            reward = 100 if next_state == goal_node else -1
            max_next_q = max([q_table[next_state][m] for m, _ in ke(next_state)], default=0)
            q_table[current_state][action] += alpha * (reward + gamma * max_next_q - q_table[current_state][action])
            current_state = next_state
        if epsilon > epsilon_end:
            epsilon *= decay


    path = ""
    current_state = start_node
    visited = {current_state}
    for _ in range(100):
        if current_state == goal_node:
            break
        valid_moves = ke(current_state)
        q_vals = [(move, q_table[current_state][move]) for move, _ in valid_moves]
        if not q_vals:
            break
        max_q = max([q for _, q in q_vals])
        best_moves = [mv for mv, q in q_vals if q == max_q]
        move = random.choice(best_moves)
        next_state = dict(valid_moves)[move]
        if next_state in visited:
            break
        path += move
        visited.add(next_state)
        current_state = next_state

    return path, time.perf_counter() - start_time


TILE_SIZE = 100
MARGIN = 5
WIDTH = HEIGHT = TILE_SIZE * 3 + MARGIN * 4 + 80 
BLUE = (30, 144, 255)       
LIGHT_BLUE = (135, 206, 250) 
DARK_GREEN = (0, 100, 0)      
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw_board(screen, state: State, highlight: int = -1):
    screen.fill(WHITE)
    for i, val in enumerate(state):
        r, c = divmod(i, 3)
        rect = pygame.Rect(
            MARGIN + c * (TILE_SIZE + MARGIN),
            MARGIN + r * (TILE_SIZE + MARGIN),
            TILE_SIZE,
            TILE_SIZE
        )
        color = LIGHT_BLUE if i == highlight and val != 0 else BLUE if val != 0 else WHITE
        pygame.draw.rect(screen, color, rect, border_radius=10) 
        if val != 0:
            font = pygame.font.Font(None, 72)
            text = font.render(str(val), True, WHITE)
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)


def draw_start_button(screen):
    button_rect = pygame.Rect(50, HEIGHT - 70, 100, 50)
    pygame.draw.rect(screen, DARK_GREEN, button_rect, border_radius=10)  
    text = font.render("Start", True, WHITE)
    text_rect = text.get_rect(center=button_rect.center)
    screen.blit(text, text_rect)
    return button_rect


def draw_runtime(screen, runtime: float):
    font = pygame.font.Font(None, 30)
    text = font.render(f"Runtime: {runtime:.2f} sec", True, BLACK)
    screen.blit(text, (160, HEIGHT - 50))


def animate_solution(screen, start_state: State, path: str, runtime: float):
    current_state = start_state
    draw_board(screen, current_state)
    draw_start_button(screen)
    draw_runtime(screen, runtime)  
    pygame.display.flip()
    time.sleep(1)

    for move in path:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Find the moving tile
        next_state = apply_move(current_state, move)
        moving_tile = [
            i for i in range(9)
            if current_state[i] != 0 and current_state[i] != next_state[i]
        ]
        highlight = moving_tile[0] if moving_tile else -1

        current_state = next_state
        draw_board(screen, current_state, highlight=highlight)
        draw_start_button(screen)
        draw_runtime(screen, runtime) 
        pygame.display.flip()
        time.sleep(0.5)

# Apply a move to the state
def apply_move(state: State, move: str) -> State:
    for m, next_state in ke(state):
        if m == move:
            return next_state
    return state

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("8 Puzzle - Q-Learning")
    clock = pygame.time.Clock()
    start_state = (2, 6, 5, 1, 3, 8, 4, 7, 0)
    path = None
    runtime = 0
    has_run = False

    # Draw initial state
    draw_board(screen, start_state)
    draw_start_button(screen)
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if draw_start_button(screen).collidepoint(event.pos) and not has_run:
                    path, runtime = q_learning(start_state)
                    has_run = True
                    animate_solution(screen, start_state, path, runtime)

        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()