import cv2

from pose_detector import CONNECTIONS, KNEE

GREEN, RED, YELLOW, WHITE, BLACK = (0, 200, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), (0, 0, 0)


def _pixel(point):
    return int(round(point[0])), int(round(point[1]))


def draw_points(frame, points):
    for point in points:
        cv2.circle(frame, _pixel(point), 4, RED, -1, cv2.LINE_AA)


def draw_skeleton(frame, points):
    for i, j in CONNECTIONS:
        cv2.line(frame, _pixel(points[i]), _pixel(points[j]), GREEN, 2, cv2.LINE_AA)
    draw_points(frame, points)


def draw_knee_angle(frame, points, side, value):
    """Evidentiaza soldul, genunchiul si glezna si scrie unghiul langa genunchi."""
    hip, knee, ankle = (_pixel(points[i]) for i in KNEE[side])
    cv2.line(frame, hip, knee, YELLOW, 4, cv2.LINE_AA)
    cv2.line(frame, knee, ankle, YELLOW, 4, cv2.LINE_AA)
    draw_text(frame, f"{value:.0f} deg", (knee[0] + 12, knee[1]))


def draw_text(frame, text, position):
    # contur negru, ca textul sa se vada pe orice fundal
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 4, cv2.LINE_AA)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
