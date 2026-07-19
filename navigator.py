# -*- coding: utf-8 -*-
"""
Deterministička navigacija kroz meni igre (template matching).

Meni je UI bez neizvesnosti pa je skriptirana navigacija pouzdanija od
učenja. Šabloni se traže multi-scale pretragom (bot/vision/template_match)
jer su slike u assets/ snimljene na jednoj razmeri, a igra na drugoj mašini
može renderovati krupnije/sitnije; pronađena razmera se pamti u
calibration.json i koristi za sve šablone, uključujući detekciju
pobede/poraza tokom partije.
"""

import os
import time

import cv2
import numpy as np
import mss
import pyautogui

from bot.vision.template_match import (
    find_template, save_template_scale, load_template_scale,
)


class PvZNavigator:
    def __init__(self):
        self.sct = mss.mss()
        self.templates = {}
        self.scale = None  # pronađena razmera šablona (None = još se traži)

        saved = load_template_scale(default=None)
        if saved is not None:
            self.scale = saved
            print(f"[NAVIGATOR] Koristim sačuvanu skalu šablona: {saved:.2f}x")

        image_names = {
            'main_menu_minigames': 'assets/btn_minigames.png',
            'icon_whack': 'assets/icon_whack.png',
            'ingame_check': 'assets/ingame_check.png',
            'click_to_start': 'assets/click_to_start.png',
            'btn_new_game_big': 'assets/btn_new_game.png',
            'btn_new_game_small': 'assets/btn_new_game_small.png',
            'btn_confirm': 'assets/btn_confirm_new_game.png'
        }
        for key, path in image_names.items():
            if os.path.exists(path):
                self.templates[key] = cv2.imread(path, cv2.IMREAD_COLOR)

    def capture_screen(self):
        monitor = self.sct.monitors[1]
        img = np.array(self.sct.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # ------------------------------------------------------------------ traženje
    def _locate(self, template_key, threshold, frame=None):
        """Vraća (cx, cy) ili None; pri prvom uspehu otkriva i pamti razmeru.

        Kada je razmera jednom poznata, koristi se ISKLJUČIVO ona - multi-scale
        fallback bi mogao da upari šablon jednog dugmeta sa sličnim dugmetom
        druge veličine i tako pokvari keširanu razmeru."""
        if template_key not in self.templates:
            return None, 0.0
        if frame is None:
            frame = self.capture_screen()
        tpl = self.templates[template_key]

        if self.scale is not None:
            found, cx, cy, _, val = find_template(frame, tpl, threshold, scale=self.scale)
            return ((cx, cy), val) if found else (None, val)

        found, cx, cy, s, val = find_template(frame, tpl, threshold)
        if found:
            self.scale = s
            save_template_scale(s)
            print(f"[NAVIGATOR] Detektovana skala šablona: {s:.2f}x (sačuvano)")
            return (cx, cy), val
        return None, val

    def click_human(self, template_key, threshold=0.75, speed=0.5):
        loc, _ = self._locate(template_key, threshold)
        if loc is None:
            return False
        cx, cy = loc
        print(f"[NAVIGATOR] Ljudski potez na: {template_key}")
        pyautogui.moveTo(cx, cy, duration=speed, tween=pyautogui.easeOutQuad)
        pyautogui.click()
        return True

    # ------------------------------------------------------------------ koraci
    def prepare_game(self, timeout=35.0):
        """Od klika na ikonu mini-igre do spremne partije: petlja stanja
        koja rešava šta god se pojavi na ekranu.

        Redosled je bitan: popup dugmad imaju prioritet, a kontrola se
        predaje tek kada je HUD vidljiv BEZ ijednog popupa u dva uzastopna
        frejma (HUD je vidljiv i iza popupa, pa sam HUD nije dovoljan!)."""
        print("[NAVIGATOR] Čekam učitavanje i rešavam dijaloge...")
        start = time.time()
        stable_frames = 0
        new_game_clicks = 0
        last_click_time = None
        last_status = 0.0

        while time.time() - start < timeout:
            now = time.time()
            frame = self.capture_screen()

            # Dijaloga ima NAJVIŠE dva ("CONTINUE GAME?" pa "NEW GAME?"
            # potvrda); posle drugog klika se više NE traže - dalje traženje
            # rizikuje lažna poklapanja na sličnim kamenim dugmadima.
            if new_game_clicks < 2:
                for key, label in (('btn_confirm', 'New Game (potvrda)'),
                                   ('btn_new_game_big', 'New Game (Continue dijalog)')):
                    loc, _ = self._locate(key, 0.80, frame=frame)
                    if loc is not None:
                        self._click_at(loc, label)
                        new_game_clicks += 1
                        last_click_time = now
                        stable_frames = 0
                        time.sleep(0.6)
                        break
                else:
                    loc = None
                if loc is not None:
                    continue

            # natpis "Click to start!" (ako postoji)
            loc, _ = self._locate('click_to_start', 0.70, frame=frame)
            if loc is not None:
                self._click_at(loc, 'click_to_start')
                time.sleep(0.5)
                return self._start_and_handoff()

            # HUD vidljiv u 2 uzastopna frejma -> partija već ide
            loc, ingame_val = self._locate('ingame_check', 0.70, frame=frame)
            if loc is not None:
                stable_frames += 1
                if stable_frames >= 2:
                    return self._start_and_handoff()
            else:
                stable_frames = 0

            # posle oba New Game klika igra UVEK kreće - kratka tišina i idemo
            if new_game_clicks >= 2 and now - last_click_time >= 2.0:
                print("[NAVIGATOR] Oba New Game klika prošla - partija kreće.")
                return self._start_and_handoff()

            # nema sačuvane partije -> nema dijaloga uopšte: posle 10 s
            # tišine pretpostavi da je nivo učitan
            if new_game_clicks == 0 and now - start >= 10.0:
                print("[NAVIGATOR] Nema dijaloga 10 s - pretpostavljam da je nivo učitan.")
                return self._start_and_handoff()

            if now - last_status >= 3.0:
                last_status = now
                print(f"[NAVIGATOR] ... čekam (klikova {new_game_clicks}/2; "
                      f"HUD marker {ingame_val:.2f})")

            time.sleep(0.35)

        print("[NAVIGATOR] Isteklo čekanje na start igre.")
        return False

    def _start_and_handoff(self):
        """Klik na sredinu travnjaka (pokreće partiju ako 'Click to start!'
        natpis stoji; bezopasan ako je već krenula) i predaja kontrole."""
        sw, sh = pyautogui.size()
        print("[NAVIGATOR] Igra spremna - klik za start i predajem kontrolu.")
        pyautogui.moveTo(int(sw * 0.5), int(sh * 0.55),
                         duration=0.4, tween=pyautogui.easeOutQuad)
        pyautogui.click()
        time.sleep(0.8)
        return True

    def _click_at(self, loc, label):
        cx, cy = loc
        print(f"[NAVIGATOR] Ljudski potez na: {label}")
        pyautogui.moveTo(cx, cy, duration=0.35, tween=pyautogui.easeOutQuad)
        pyautogui.click()

    def _click_with_retry(self, template_key, attempts=3, wait_between=1.0, speed=0.6):
        """Meni ume da se učitava/animira - probaj nekoliko puta pre odustajanja."""
        best_val = 0.0
        for i in range(attempts):
            loc, val = self._locate(template_key, threshold=0.75)
            best_val = max(best_val, val)
            if loc is not None:
                cx, cy = loc
                print(f"[NAVIGATOR] Ljudski potez na: {template_key}")
                pyautogui.moveTo(cx, cy, duration=speed, tween=pyautogui.easeOutQuad)
                pyautogui.click()
                return True
            if i < attempts - 1:
                print(f"[NAVIGATOR] '{template_key}' još nije vidljiv "
                      f"(najbolje poklapanje {best_val:.2f}), čekam...")
                time.sleep(wait_between)
        print(f"[NAVIGATOR] '{template_key}' NIJE pronađen ni multi-scale pretragom "
              f"(najbolje poklapanje {best_val:.2f}, prag 0.75).")
        print("[NAVIGATOR] Ako je igra vidljiva na ekranu, šablon iz assets/ se "
              "razlikuje od izgleda na ovoj mašini - treba ga ponovo snimiti.")
        return False

    def start_whack_a_zombie(self):
        if not self._click_with_retry('main_menu_minigames'):
            return False

        time.sleep(1.0)

        if not self._click_with_retry('icon_whack'):
            return False

        return self.prepare_game()
