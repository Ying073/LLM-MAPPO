import numpy as np

def reward(env, n, action, prev_au):
    # Current position and next position
    pos = env.uav_pos[n].copy()
    next_pos = pos.copy()
    if action == 0: next_pos[1] += 1  # north
    elif action == 1: next_pos[1] -= 1  # south
    elif action == 2: next_pos[0] += 1  # east
    elif action == 3: next_pos[0] -= 1  # west
    elif action == 4: next_pos[2] += 1  # ascend
    elif action == 5: next_pos[2] -= 1  # descend

    # Hard constraint penalties (invalid moves)
    if next_pos[0] < 0 or next_pos[0] >= LX or next_pos[1] < 0 or next_pos[1] >= LY:
        return -10.0
    if next_pos[2] < 1 or next_pos[2] > 5:
        return -10.0
    # Obstacle collision
    if env.occ[next_pos[0], next_pos[1]] == 1:
        return -10.0
    # Inter-UAV separation
    for m in range(N_UAV):
        if m != n:
            dist = np.linalg.norm(next_pos[:2] - env.uav_pos[m][:2])
            if dist < 1.0:  # d_min = 1 grid cell
                return -10.0

    # Altitude-dependent sensing parameters
    alt = next_pos[2]
    sensing_range = 1 + alt  # higher altitude → larger range
    detect_prob = 0.9 - 0.1 * alt  # lower altitude → higher detection
    false_alarm = 0.05 * alt

    r = 0.0
    # Target confirmation reward (highest priority)
    newly_confirmed = 0
    for dx in range(-sensing_range, sensing_range+1):
        for dy in range(-sensing_range, sensing_range+1):
            if abs(dx) + abs(dy) > sensing_range:
                continue
            cx = next_pos[0] + dx
            cy = next_pos[1] + dy
            if 0 <= cx < LX and 0 <= cy < LY and env.occ[cx, cy] == 0:
                if env.gtpm[cx, cy] >= TARGET_CONFIRM_THRESHOLD and env.searched[cx, cy] == 0:
                    newly_confirmed += 1
                    r += 5.0  # strong reward for confirming target
    r += 0.5 * newly_confirmed  # extra for multiple

    # Uncertainty reduction (area coverage)
    au_before = prev_au
    # Simulate uncertainty reduction after action (simplified)
    au_after = au_before
    for dx in range(-sensing_range, sensing_range+1):
        for dy in range(-sensing_range, sensing_range+1):
            if abs(dx) + abs(dy) > sensing_range:
                continue
            cx = next_pos[0] + dx
            cy = next_pos[1] + dy
            if 0 <= cx < LX and 0 <= cy < LY and env.occ[cx, cy] == 0:
                au_after -= 0.01 * (1.0 - env.gtpm[cx, cy])  # reduce uncertainty
    r += (au_before - au_after) * 2.0  # encourage uncertainty reduction

    # Descending reward when near potential target (detected but unconfirmed)
    potential_target_near = False
    for dx in range(-sensing_range, sensing_range+1):
        for dy in range(-sensing_range, sensing_range+1):
            if abs(dx) + abs(dy) > sensing_range:
                continue
            cx = next_pos[0] + dx
            cy = next_pos[1] + dy
            if 0 <= cx < LX and 0 <= cy < LY:
                if env.gtpm[cx, cy] > 0.5 and env.searched[cx, cy] == 0:
                    potential_target_near = True
    if potential_target_near and action == 5:  # descend
        r += 1.0
    elif potential_target_near and action == 4:  # ascend away from target
        r -= 0.5

    # Dispersal encouragement (coverage diversity)
    min_dist_to_others = min(np.linalg.norm(next_pos[:2] - env.uav_pos[m][:2]) 
                             for m in range(N_UAV) if m != n)
    if min_dist_to_others > 3.0:
        r += 0.2
    elif min_dist_to_others < 2.0:
        r -= 0.1

    # Small step penalty to avoid oscillation
    r -= 0.01
    return r