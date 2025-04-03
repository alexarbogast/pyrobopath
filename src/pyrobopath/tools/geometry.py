
def orientation(p, q, r, tol=10e-2):
    """Returns true if p, q, r is CW, false if CCW"""
    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if val > tol:  # cw
        return 1
    elif val < -tol:  # ccw
        return 2
    else:  # collinear
        return 0

# Given three collinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def on_segment(p, q, r):
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
           (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False
