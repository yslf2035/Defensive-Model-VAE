
def Reg157(x_ego, v_ego, x_front, v_front):
    v_rel = v_ego - v_front
    ttc = abs(x_front - x_ego) / v_rel
    threshold = v_rel / (2 * 6) + 0.35
    if ttc > threshold:
        return -6
    else:
        return None