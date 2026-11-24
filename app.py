"""Stage 1: schelet si unghiul genunchiului, din camera sau dintr-un clip.

    python app.py                        camera
    python app.py videos/clip.mp4        clip
"""
import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # graficul se salveaza, nu se afiseaza
import matplotlib.pyplot as plt

from drawing import draw_knee_angle, draw_points, draw_skeleton, draw_text
from geometry import angle
from pose_detector import KNEE, PoseDetector

MIN_VISIBILITY = 0.5
MAX_SIZE = 960  # latura maxima a cadrului procesat, in pixeli
WINDOW = "Contor exercitii - stage 1 (q = iesire)"


def resize(frame):
    h, w = frame.shape[:2]
    scale = MAX_SIZE / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def knee_angle(points, side):
    """Unghiul genunchiului sau None, daca un punct nu se vede destul de bine."""
    leg = points[list(KNEE[side])]
    if leg[:, 2].min() < MIN_VISIBILITY:
        return None
    value = angle(*leg[:, :2])
    return None if math.isnan(value) else value


class Recorder:
    """Retine masuratorile si cadrele cu unghiul minim si maxim, pe fiecare picior."""

    def __init__(self):
        self.times = []
        self.angles = {side: [] for side in KNEE}
        self.visibility = {side: 0.0 for side in KNEE}
        self.lowest = {side: None for side in KNEE}
        self.highest = {side: None for side in KNEE}

    @property
    def side(self):
        """Piciorul vazut cel mai bine pana acum (din lateral, celalalt e acoperit)."""
        return max(self.visibility, key=self.visibility.get)

    def add(self, seconds, frame, points):
        self.times.append(seconds)
        for side in KNEE:
            value = None if points is None else knee_angle(points, side)
            self.angles[side].append(value)
            if points is not None:
                self.visibility[side] += points[list(KNEE[side]), 2].mean()
            if value is None:
                continue
            if self.lowest[side] is None or value < self.lowest[side][0]:
                self.lowest[side] = (value, frame.copy(), points)
            if self.highest[side] is None or value > self.highest[side][0]:
                self.highest[side] = (value, frame.copy(), points)

    def save(self, folder):
        """Salveaza figurile pentru documentatie. Intoarce False daca nu exista unghiuri."""
        side = self.side
        if self.lowest[side] is None:
            return False
        folder.mkdir(parents=True, exist_ok=True)
        _, frame, points = self.lowest[side]
        write(folder / "1_sursa.png", frame)
        with_points = frame.copy()
        draw_points(with_points, points)
        write(folder / "2_puncte.png", with_points)
        for name, (value, frame, points) in (("3a_jos", self.lowest[side]), ("3b_sus", self.highest[side])):
            image = frame.copy()
            draw_skeleton(image, points)
            draw_knee_angle(image, points, side, value)
            write(folder / f"{name}.png", image)
        self._save_plot(folder / "4_grafic_unghi.png", side)
        with open(folder / "unghiuri.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["secunde", "unghi_grade"])
            for t, value in zip(self.times, self.angles[side]):
                writer.writerow([f"{t:.3f}", "" if value is None else f"{value:.2f}"])
        return True

    def _save_plot(self, path, side):
        # None devine NaN: graficul se intrerupe unde piciorul nu se vede
        values = [math.nan if v is None else v for v in self.angles[side]]
        name = "stâng" if side == "left" else "drept"
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(self.times, values)
        ax.set_xlabel("Timp [s]")
        ax.set_ylabel("Unghiul genunchiului [°]")
        ax.set_title(f"Unghiul genunchiului {name} în timp")
        ax.set_ylim(0, 190)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def write(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Nu pot salva {path}")


def run(source, show=True):
    is_camera = source.isdigit()
    if not is_camera and not Path(source).is_file():
        raise SystemExit(f"Clipul nu exista: {source}")
    capture = cv2.VideoCapture(int(source), cv2.CAP_DSHOW) if is_camera else cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Nu pot deschide sursa: {source}")
    video_fps = capture.get(cv2.CAP_PROP_FPS) or 30

    recorder = Recorder()
    frames = detected = 0
    processing = 0.0  # secunde petrecute in detectie si desenare, fara citire si afisare
    timestamp = -1
    start = time.perf_counter()
    try:
        with PoseDetector() as detector:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                seconds = time.perf_counter() - start if is_camera else frames / video_fps
                timestamp = max(timestamp + 1, int(seconds * 1000))

                begin = time.perf_counter()
                frame = resize(frame)
                points = detector.detect(frame, timestamp)
                output = frame.copy()
                if points is not None:
                    detected += 1
                    draw_skeleton(output, points)
                processing += time.perf_counter() - begin
                frames += 1

                recorder.add(seconds, frame, points)
                value = recorder.angles[recorder.side][-1]
                if value is not None:
                    draw_knee_angle(output, points, recorder.side, value)
                draw_text(output, f"FPS: {frames / processing:.0f}", (10, 30))
                if show:
                    cv2.imshow(WINDOW, output)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()

    if frames == 0:
        raise SystemExit("Nu am citit niciun cadru.")
    print(f"Cadre: {frames}, cu persoana detectata: {detected} ({100 * detected / frames:.0f}%)")
    print(f"Viteza de procesare: {frames / processing:.1f} FPS ({1000 * processing / frames:.1f} ms/cadru)")
    values = [v for v in recorder.angles[recorder.side] if v is not None]
    print(f"Cadre cu unghi masurat: {len(values)}")
    folder = Path(__file__).parent / "results" / ("camera" if is_camera else Path(source).stem)
    if recorder.save(folder):
        print(f"Unghi minim {min(values):.1f}, maxim {max(values):.1f} grade (piciorul: {recorder.side})")
        print(f"Figuri salvate in {folder}")
    else:
        print("Piciorul nu s-a vazut destul de bine, nu am ce figuri sa salvez.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: schelet si unghiul genunchiului")
    parser.add_argument("source", nargs="?", default="0", help="numarul camerei (implicit 0) sau calea clipului")
    parser.add_argument("--no-display", action="store_true", help="proceseaza fara fereastra")
    args = parser.parse_args()
    run(args.source, show=not args.no_display)
