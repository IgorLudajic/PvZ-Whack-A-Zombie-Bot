# -*- coding: utf-8 -*-
"""
Petlja prave partije: kaptura ekrana -> percepcija (YOLO + HSV + HUD) ->
Snapshot -> RL politika -> humanizovano izvršenje. Vrti se dok partija
ne bude dobijena ili izgubljena.
"""

import os
import time
from datetime import datetime

import cv2

import config
from bot.actions import decode
from bot.state import encode_observation, build_action_mask
from bot.sim.simulator import EpisodeStats
from bot.vision.capture import ScreenCapture
from bot.vision.detector import YoloDetector
from bot.vision.sun_finder import find_suns
from bot.vision.hud_reader import HudReader
from bot.vision.mower_watcher import MowerWatcher
from bot.vision.tracker import TargetTracker
from bot.game.state_builder import GridMapper, build_snapshot
from bot.control.humanizer import Humanizer
from bot.control.executor import ActionExecutor

NOMINAL_LEVEL_DURATION = 230.0  # s - za time_progress komponentu stanja


class RealGameSession:
    def __init__(self, policy):
        self.policy = policy
        self.screen = ScreenCapture()
        self.detector = YoloDetector()
        self.hud = HudReader(self.screen)
        self.mowers = MowerWatcher(self.screen)
        self.tracker = TargetTracker()
        self.mapper = GridMapper(self.screen)
        self.humanizer = Humanizer(self.screen.bbox)
        self.stats = EpisodeStats()
        self.executor = ActionExecutor(
            self.humanizer, self.hud, self.mapper, self.tracker, self.stats)

        self.tpl_win = cv2.imread(config.TEMPLATE_WIN)
        self.tpl_loss = cv2.imread(config.TEMPLATE_LOSS)
        self._danger_seen = set()

        self._video = None
        if config.DEBUG_RECORD_VIDEO:
            os.makedirs(config.STATS_DIR, exist_ok=True)
            path = os.path.join(
                config.STATS_DIR,
                f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            self._video = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), 5,
                (self.screen.width // 2, self.screen.height // 2))
            print(f"[DEBUG] Snimam pregled partije u {path}")

    # ---------------------------------------------------------------- helpers
    def _match(self, frame, template):
        if template is None:
            return None
        res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= config.TEMPLATE_MATCH_THRESHOLD:
            h, w = template.shape[:2]
            return (max_loc[0] + w // 2, max_loc[1] + h // 2)
        return None

    def _update_danger_stats(self, now):
        for tr in self.tracker.alive(now):
            cf = self.mapper.col_float(tr.cx)
            self.stats.closest_approach = min(self.stats.closest_approach, cf)
            if cf < 1.0 and tr.id not in self._danger_seen:
                self._danger_seen.add(tr.id)
                self.stats.danger_near_count += 1

    def _record_debug(self, frame, action_desc):
        if self._video is None:
            return
        small = cv2.resize(frame, (self.screen.width // 2, self.screen.height // 2))
        cv2.putText(small, action_desc, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        self._video.write(small)

    # ------------------------------------------------------------------- play
    def play(self):
        """Igra partiju do kraja; vraća EpisodeStats."""
        print("[AGENT] Partija počinje - RL politika preuzima kontrolu.")
        start = time.time()
        frame_idx = 0

        try:
            while True:
                now = time.time()
                elapsed = now - start
                if elapsed > config.GAME_HARD_TIMEOUT:
                    self.stats.result = "TIMEOUT"
                    break

                frame = self.screen.grab_bgr()
                frame_idx += 1

                # kraj partije? (proveravamo svaki drugi frejm - matchTemplate košta)
                if frame_idx % 2 == 0:
                    win_pos = self._match(frame, self.tpl_win)
                    if win_pos is not None:
                        print("[AGENT] Pobeda! Kupim vreću sa novcem.")
                        self.humanizer.move_to(self.screen.left + win_pos[0],
                                               self.screen.top + win_pos[1])
                        self.humanizer.click()
                        self.stats.result = "WIN"
                        break
                    if self._match(frame, self.tpl_loss) is not None:
                        self.stats.result = "LOSS"
                        break

                # percepcija
                moving_dets, grave_dets = self.detector.detect(frame)
                self.tracker.update(moving_dets, now)
                suns_px = find_suns(frame)
                card_ready = self.hud.card_states(frame)
                mowers_left = self.mowers.update(frame)
                self.stats.mowers_used = self.mowers.used_count
                self._update_danger_stats(now)

                # stanje -> politika
                snapshot, cell_tracks = build_snapshot(
                    self.mapper, self.tracker.alive(now), grave_dets, suns_px,
                    card_ready, mowers_left,
                    elapsed / NOMINAL_LEVEL_DURATION,
                    self.executor.sun_ledger, self.executor.freeze_remaining, now)
                obs = encode_observation(snapshot)
                mask = build_action_mask(snapshot)
                action = self.policy.act(obs, mask)
                kind, r, c = decode(action)
                self.stats.actions[kind] = self.stats.actions.get(kind, 0) + 1
                self._record_debug(frame, f"{kind} {r},{c}" if r is not None else kind)

                # izvršenje
                self.humanizer.reaction_jitter()
                if kind == "whack":
                    self.executor.whack_cell(r, c, cell_tracks)
                elif kind == "buster":
                    self.executor.plant("gravebuster", r, c)
                elif kind == "cherry":
                    self.executor.plant("cherry", r, c)
                elif kind == "ice":
                    self.executor.plant_ice(snapshot)
                elif kind == "collect_sun":
                    self.executor.collect_sun(suns_px)
                else:
                    self.executor.wait()

        except KeyboardInterrupt:
            print("\n[AGENT] Prekinuto od strane korisnika.")
            self.stats.result = "INTERRUPTED"
        finally:
            self.stats.duration = time.time() - start
            if self._video is not None:
                self._video.release()

        return self.stats
