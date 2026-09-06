def reward(env, n, action, prev_au):
    x, y, h = (int(v) for v in env.uav_pos[n])

    nx, ny, nh = x, y, h

    if action == 0:        # North
        ny += 1
    elif action == 1:      # East
        nx += 1
    elif action == 2:      # South
        ny -= 1
    elif action == 3:      # West
        nx -= 1
    elif action == 4:      # Ascend
        nh += 1
    elif action == 5:      # Descend
        nh -= 1

    # ----- Hard constraints -----
    if not (0 <= nx < LX and 0 <= ny < LY) or nh < 0:
        return -30.0

    if env.occ[nx, ny] == 1:
        return -30.0

    MIN_SEP = 1.0
    cand_3d = np.array([nx, ny, nh], dtype=float)

    for m in range(N_UAV):
        if m == n:
            continue
        dist_3d = np.linalg.norm(env.uav_pos[m].astype(float) - cand_3d)
        if dist_3d < MIN_SEP - 1e-6:
            return -10.0

    # ----- Global uncertainty-progress baseline -----
    r = 0.0
    au_now = env.area_uncertainty()
    if prev_au is not None:
        r += 0.6 * (prev_au - au_now)
    r -= 0.08 * au_now

    # ----- Sensor footprint -----
    R = int(max(1, 1 + nh))
    det_factor = 1.0 / (1.0 + max(0, nh))

    coverage_value = 0.0
    search_value = 0.0
    target_value = 0.0
    core_target = 0.0

    x0, x1 = max(0, nx - R), min(LX, nx + R + 1)
    y0, y1 = max(0, ny - R), min(LY, ny + R + 1)

    for gi in range(x0, x1):
        for gj in range(y0, y1):
            # Obstacle cells do not contribute to searchable-area uncertainty
            if env.occ[gi, gj] > 0.5:
                continue

            p = float(env.gtpm[gi, gj])
            u = float(env.geum[gi, gj])
            ever_searched = float(env.searched[gi, gj]) > 0.5

            # Direct signal to reduce remaining environment uncertainty
            coverage_value += u

            # Probability-weighted uncertainty; avoid chasing already confirmed targets forever
            info_gain = p * u
            if ever_searched:
                info_gain *= 0.05
            search_value += info_gain

            # Target confirmation reward only until it is actually confirmed/searched
            if env.zeta[gi, gj] > 0.5:
                target_value += 0.05 if ever_searched else 1.0

            if p >= TARGET_CONFIRM_THRESHOLD and not ever_searched:
                target_value += 0.5

            # Small verification neighbourhood directly under UAV
            if abs(gi - nx) <= 1 and abs(gj - ny) <= 1:
                if env.zeta[gi, gj] > 0.5:
                    core_target += 0.05 if ever_searched else 1.0
                if p >= TARGET_CONFIRM_THRESHOLD and not ever_searched:
                    core_target += 0.5

    # ----- Area-coverage reward -----
    r += 0.55 * coverage_value * det_factor

    # ----- Target-probability uncertainty reduction -----
    r += 0.50 * search_value * det_factor

    # ----- Target-capture priority -----
    r += 12.0 * target_value * det_factor

    # ----- Descend to verify a likely nearby target -----
    if action == 5:
        r += 1.2 * core_target * det_factor

    # Ascending away from a likely target is discouraged
    if action == 4:
        r -= 0.5 * core_target

    # ----- Altitude/energy shaping -----
    r -= 0.04 * nh * nh

    # ----- Dispersion among UAVs -----
    cand_2d = np.array([nx, ny], dtype=float)
    xy_dist = np.linalg.norm(env.uav_pos[:, :2].astype(float) - cand_2d, axis=1)
    xy_dist[n] = np.inf
    nearest_xy = float(np.min(xy_dist))
    r += 0.12 * max(0.0, nearest_xy - 1.0)

    return r