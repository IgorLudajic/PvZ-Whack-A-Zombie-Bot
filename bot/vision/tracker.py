# -*- coding: utf-8 -*-
"""
Praćenje meta kroz frejmove (stabilni identiteti).

Rešava dva problema v1.0 bota:
1. Dupli udarci: zombi u animaciji umiranja se i dalje detektuje, pa ga je
   v1.0 ponovo gađao (naduvana statistika, potrošeno vreme). Ovde meta posle
   zadnjeg udarca ulazi u "dying" period i ne nudi se politici.
2. Nedovršena ubistva: ako YOLO promeni klasu (buckethead -> damaged_
   buckethead) ili detekcija zatreperi, track zadržava identitet i pamti
   koliko je udaraca skoro zadato, pa se broj preostalih udaraca ne resetuje.
"""

import math
import time
from dataclasses import dataclass, field

# klasa detekcije -> osnovna vrsta (za statistiku) i broj potrebnih udaraca
KIND_OF_CLASS = {
    "zombie": "zombie",
    "conehead": "conehead",
    "buckethead": "buckethead",
    "damaged_buckethead": "buckethead",
}
HITS_OF_CLASS = {
    "zombie": 1,
    "conehead": 2,
    "buckethead": 3,
    "damaged_buckethead": 2,
}

MATCH_RADIUS = 90        # px - maksimalno rastojanje za uparivanje detekcije i tracka
TRACK_TTL = 0.6          # s - track bez detekcije se briše
DYING_GRACE = 0.9        # s - posle smrtonosnog udarca meta se ne nudi ponovo
PERCEPTION_LAG = 0.45    # s - koliko dugo udarac "još nije vidljiv" u percepciji


@dataclass
class Track:
    id: int
    cls_name: str
    cx: float
    cy: float
    h: float
    last_seen: float
    recent_hits: list = field(default_factory=list)  # vremena zadatih udaraca
    dying_until: float = 0.0

    @property
    def aim_px(self):
        """Tačka nišanjenja: malo ispod centra bounding box-a (telo zombija)."""
        return (self.cx, self.cy + 0.28 * self.h)

    def hits_remaining(self, now):
        base = HITS_OF_CLASS[self.cls_name]
        fresh = sum(1 for t in self.recent_hits if now - t < PERCEPTION_LAG)
        return max(base - fresh, 0)

    @property
    def kind(self):
        return KIND_OF_CLASS[self.cls_name]


class TargetTracker:
    def __init__(self):
        self.tracks = []
        self._next_id = 1

    def update(self, detections, now=None):
        """Uparuje YOLO detekcije (zombiji) sa postojećim track-ovima."""
        now = now or time.time()
        unmatched = list(self.tracks)

        for det in detections:
            if det.cls_name not in HITS_OF_CLASS:
                continue
            best, best_d = None, MATCH_RADIUS
            for tr in unmatched:
                d = math.hypot(tr.cx - det.cx, tr.cy - det.cy)
                if d < best_d:
                    best, best_d = tr, d
            if best is not None:
                unmatched.remove(best)
                best.cls_name = det.cls_name
                best.cx, best.cy = det.cx, det.cy
                best.h = det.y2 - det.y1
                best.last_seen = now
            else:
                self.tracks.append(Track(
                    id=self._next_id, cls_name=det.cls_name,
                    cx=det.cx, cy=det.cy, h=det.y2 - det.y1, last_seen=now,
                ))
                self._next_id += 1

        self.tracks = [t for t in self.tracks if now - t.last_seen <= TRACK_TTL]

    def register_hit(self, track, now=None):
        now = now or time.time()
        track.recent_hits.append(now)
        if track.hits_remaining(now) <= 0:
            track.dying_until = now + DYING_GRACE

    def alive(self, now=None):
        """Mete koje politika sme da bira (nisu u animaciji umiranja)."""
        now = now or time.time()
        return [t for t in self.tracks
                if t.dying_until <= now and t.hits_remaining(now) > 0]
