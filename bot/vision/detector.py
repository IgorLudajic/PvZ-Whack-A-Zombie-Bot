# -*- coding: utf-8 -*-
"""
YOLO detekcija sa pragom poverenja po klasi i vremenskom postojanošću
za grobove.

Grobovi su u v1.0 bili slaba tačka detekcije: statični su i delimično
zaklonjeni zombijima, pa im YOLO daje niske skorove. Rešenje: nizak prag
za klasu 'grave' + zahtev da se grob pojavi u bar N od poslednjih M
frejmova na istom polju (lažne pozitive ne preživljavaju filter, a prave
detekcije se stabilizuju).
"""

from collections import deque
from dataclasses import dataclass

from ultralytics import YOLO

import config


@dataclass
class Detection:
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2


class YoloDetector:
    def __init__(self, model_path=config.YOLO_MODEL_PATH):
        self.model = YOLO(model_path)
        self.min_conf = min(config.CLASS_CONF.values())
        self.grave_history = deque(maxlen=config.GRAVE_PERSISTENCE_WINDOW)

    def detect(self, frame_bgr):
        """Vraća (detekcije_zombija_i_bustera, potvrđeni_grobovi_kao_detekcije)."""
        results = self.model.predict(
            source=frame_bgr, conf=self.min_conf,
            imgsz=config.YOLO_IMGSZ, verbose=False,
        )
        result = results[0]

        moving, graves_now = [], []
        for box in result.boxes:
            cls_name = result.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if conf < config.CLASS_CONF.get(cls_name, 0.30):
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().numpy())
            det = Detection(cls_name, conf, x1, y1, x2, y2)
            if cls_name == "grave":
                graves_now.append(det)
            else:
                moving.append(det)

        confirmed_graves = self._confirm_graves(graves_now)
        return moving, confirmed_graves

    def _confirm_graves(self, graves_now):
        """Vremenska postojanost: grob je potvrđen ako je viđen u bar
        GRAVE_PERSISTENCE_FRAMES od poslednjih GRAVE_PERSISTENCE_WINDOW
        frejmova na približno istom mestu (poluprečnik ~45 px)."""
        self.grave_history.append(graves_now)

        confirmed = []
        for det in graves_now:
            seen = 0
            for frame_dets in self.grave_history:
                for old in frame_dets:
                    if abs(old.cx - det.cx) < 45 and abs(old.cy - det.cy) < 45:
                        seen += 1
                        break
            if seen >= config.GRAVE_PERSISTENCE_FRAMES:
                confirmed.append(det)
        return confirmed
