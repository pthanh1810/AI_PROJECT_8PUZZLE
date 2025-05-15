import pygame
import sys
import time
import random
from collections import defaultdict
from typing import Tuple, Dict, Set, List


State = Tuple[int, ...]
Action = str
GOAL: State = (1, 2, 3, 4, 5, 6, 7, 8, 0)  # Trạng thái đích

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
PRIMARY_BG = (230, 236, 239)  # #E6ECEF
SECONDARY_BG = (245, 247, 250)  # #F5F7FA
TILE_FILLED = (227, 6, 19)  # #E30613 (Vietjet Red)
TILE_HIGHLIGHT = (255, 102, 102)  # #FF6666 (Light Red)
TILE_EMPTY = (255, 255, 255)  # #FFFFFF
TILE_TEXT = (255, 255, 255)  # #FFFFFF
START_BUTTON_1 = (255, 193, 7)  # #FFC107 (Vietjet Yellow)
START_BUTTON_2 = (255, 236, 179)  # #FFECB3 (Light Yellow)
RERUN_BUTTON_1 = (227, 6, 19)  # #E30613 (Vietjet Red)
RERUN_BUTTON_2 = (255, 102, 102)  # #FF6666 (Light Red)
BUTTON_TEXT = (255, 255, 255)  # #FFFFFF
ACCENT_COLOR = (124, 58, 237)  # #7C3AED
SHADOW_COLOR = (0, 0, 0, 50)  # Semi-transparent black


def draw_gradient_rect(screen, rect, color1, color2):
    x, y, w, h = rect
    shadow_surface = pygame.Surface((w + 4, h + 4), pygame.SRCALPHA)
    pygame.draw.rect(shadow_surface, SHADOW_COLOR, (2, 2, w, h), border_radius=12)
    screen.blit(shadow_surface, (x - 2, y - 2))
    gradient_surface = pygame.Surface((w, h))
    for i in range(h):
        ratio = i / h
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        pygame.draw.line(gradient_surface, (r, g, b), (0, i), (w, i))
    screen.blit(gradient_surface, (x, y))


def draw_board(screen, state: State, tile_font, highlight: int = -1):
    draw_gradient_rect(screen, pygame.Rect(0, 0, WIDTH, HEIGHT), PRIMARY_BG, SECONDARY_BG)
    shadow_surface = pygame.Surface((TILE_SIZE * 3 + MARGIN * 2 + 10, TILE_SIZE * 3 + MARGIN * 2 + 10), pygame.SRCALPHA)
    pygame.draw.rect(shadow_surface, SHADOW_COLOR, (5, 5, TILE_SIZE * 3 + MARGIN * 2, TILE_SIZE * 3 + MARGIN * 2), border_radius=15)
    screen.blit(shadow_surface, (MARGIN - 5, MARGIN - 5))
    
    for i, val in enumerate(state):
        r, c = divmod(i, 3)
        rect = pygame.Rect(
            MARGIN + c * (TILE_SIZE + MARGIN),
            MARGIN + r * (TILE_SIZE + MARGIN),
            TILE_SIZE,
            TILE_SIZE
        )
        color = TILE_HIGHLIGHT if i == highlight and val != 0 else TILE_FILLED if val != 0 else TILE_EMPTY
        pygame.draw.rect(screen, color, rect, border_radius=10)
        pygame.draw.rect(screen, ACCENT_COLOR, rect, 2, border_radius=10)
        if val != 0:
            text = tile_font.render(str(val), True, TILE_TEXT)
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)


def draw_start_button(screen, button_font):
    button_rect = pygame.Rect(50, HEIGHT - 70, 100, 50)
    mouse_pos = pygame.mouse.get_pos()
    color1 = START_BUTTON_1 if not button_rect.collidepoint(mouse_pos) else START_BUTTON_2
    color2 = START_BUTTON_2 if not button_rect.collidepoint(mouse_pos) else START_BUTTON_1
    draw_gradient_rect(screen, button_rect, color1, color2)
    pygame.draw.rect(screen, ACCENT_COLOR, button_rect, 2, border_radius=10)
    text = button_font.render("Start", True, BUTTON_TEXT)
    text_rect = text.get_rect(center=button_rect.center)
    screen.blit(text, text_rect)
    return button_rect




def draw_runtime(screen, runtime: float, button_font):
    text = button_font.render(f"Runtime: {runtime:.2f} sec", True, (0, 0, 0))
    screen.blit(text, (160, HEIGHT - 50))


def animate_solution(screen, start_state: State, path: str, runtime: float, tile_font, button_font):
    current_state = start_state
    draw_board(screen, current_state, tile_font)
    draw_start_button(screen, button_font)
    draw_runtime(screen, runtime, button_font)
    pygame.display.flip()
    time.sleep(1)

    for move in path:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        next_state = apply_move(current_state, move)
        moving_tile = [
            i for i in range(9)
            if current_state[i] != 0 and current_state[i] != next_state[i]
        ]
        highlight = moving_tile[0] if moving_tile else -1

        current_state = next_state
        draw_board(screen, current_state, tile_font, highlight=highlight)
        draw_start_button(screen, button_font)
        draw_runtime(screen, runtime, button_font)
        pygame.display.flip()
        time.sleep(0.5)


def apply_move(state: State, move: str) -> State:
    for m, next_state in ke(state):
        if m == move:
            return next_state
    return state


def main():
    pygame.init()
    if not pygame.font.get_init():
        print("Pygame font module failed to initialize!")
        pygame.quit()
        sys.exit(1)

    try:
        tile_font = pygame.font.Font(None, 72)
        button_font = pygame.font.Font(None, 36)
    except Exception as e:
        print(f"Failed to initialize fonts: {e}")
        pygame.quit()
        sys.exit(1)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("8 Puzzle - Q-Learning")
    clock = pygame.time.Clock()
    start_state = (2, 6, 5, 1, 3, 8, 4, 7, 0)
    path = None
    runtime = 0
    has_run = False

    running = True
    draw_board(screen, start_state, tile_font)
    draw_start_button(screen, button_font)
    pygame.display.flip()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if draw_start_button(screen, button_font).collidepoint(event.pos) and not has_run:
                    path, runtime = q_learning(start_state)
                    has_run = True
                    animate_solution(screen, start_state, path, runtime, tile_font, button_font)
        draw_board(screen, start_state, tile_font)
        draw_start_button(screen, button_font)
        draw_runtime(screen, runtime, button_font)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()