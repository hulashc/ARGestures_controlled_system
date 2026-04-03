"""Utility functions for smoothing, debouncing, and visual effects."""

import numpy as np
import random


class Smoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, raw_value):
        if self.value is None:
            self.value = np.array(raw_value, dtype=float)
            return tuple(self.value)
        raw = np.array(raw_value, dtype=float)
        self.value = self.alpha * raw + (1 - self.alpha) * self.value
        return tuple(self.value)

    def reset(self):
        self.value = None


class LandmarkSmoother:
    def __init__(self, alpha=0.3):
        self.smoothers = [Smoother(alpha=alpha) for _ in range(21)]

    def update(self, landmarks):
        if landmarks is None:
            self.reset()
            return None
        return [self.smoothers[i].update(lm) for i, lm in enumerate(landmarks)]

    def reset(self):
        for s in self.smoothers:
            s.reset()


class CursorSmoother:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.value = None

    def update(self, x, y):
        if self.value is None:
            self.value = np.array([x, y], dtype=float)
            return int(x), int(y)
        raw = np.array([x, y], dtype=float)
        self.value = self.alpha * raw + (1 - self.alpha) * self.value
        return int(self.value[0]), int(self.value[1])

    def reset(self):
        self.value = None


class GestureDebounce:
    def __init__(self, hold_frames=5):
        self.hold_frames = hold_frames
        self.current_gesture = "none"
        self.streak = 0
        self.fired = False

    def update(self, raw_gesture):
        if raw_gesture == self.current_gesture:
            self.streak += 1
        else:
            self.current_gesture = raw_gesture
            self.streak = 1
            self.fired = False

        if self.streak >= self.hold_frames and not self.fired:
            self.fired = True
            return self.current_gesture
        return None

    def reset(self):
        self.current_gesture = "none"
        self.streak = 0
        self.fired = False


class Particle:
    def __init__(self, x, y, color):
        self.x = float(x)
        self.y = float(y)
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-3, 3)
        self.life = random.randint(15, 30)
        self.max_life = self.life
        self.color = color
        self.size = random.randint(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # gravity
        self.life -= 1
        return self.life > 0

    def draw(self, frame):
        alpha = self.life / self.max_life
        c = tuple(int(ch * alpha) for ch in self.color)
        import cv2
        cv2.circle(frame, (int(self.x), int(self.y)), self.size, c, -1)


class ParticleSystem:
    def __init__(self):
        self.particles = []

    def emit(self, x, y, color, count=20):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))

    def update_and_draw(self, frame):
        self.particles = [p for p in self.particles if p.update()]
        for p in self.particles:
            p.draw(frame)
