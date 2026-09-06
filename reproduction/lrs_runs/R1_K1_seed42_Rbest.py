def reward(env, n, action, prev_au):
    x = int(env.uav_pos[n, 0])
    y = int(env.uav_pos[n, 1])
    alt = float(env.uav_pos[n, 2])

    if not (0 <= x < LX and 0 <= y < LY) or env.occ[x, y]:
        return -10.0

    # Target discovery: reward only newly confirmed searched cells.
    s = int(np.sum(env.searched))
    prev_s = getattr(reward, "_prev_searched_sum", 0)
    if s < prev_s:          # episode reset
        prev_s = 0
    r = 10.0 * float(s - prev_s)
    reward._prev_searched_sum = s

    # Dense uncertainty reduction reward; altitude improvements enter through this.
    au = env.area_uncertainty()
    du = prev_au - au
    if du > 0.0:
        r += 20.0 * du

    # Collision penalty with another UAV at same 3D cell.
    pos = env.uav_pos
    o = np.delete(pos, n, axis=0)
    if np.any(
        (o[:, 0].astype(int) == x)
        & (o[:, 1].astype(int) == y)
        & (np.abs(o[:, 2] - alt) < 0.5)
    ):
        r -= 5.0

    # Separation: discourage same/near horizontal cells.
    d = np.hypot(o[:, 0] - x, o[:, 1] - y)
    if d.size:
        md = np.min(d)
        if md < 1.0:
            r -= 1.0
        elif md < 3.0:
            r -= 0.2 * (3.0 - md)
        else:
            r += 0.02

    return float(r)