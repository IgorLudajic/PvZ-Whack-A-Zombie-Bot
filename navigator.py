import cv2
import numpy as np
import pyautogui
import mss
import time
import os

class PvZNavigator:
    def __init__(self):
        self.sct = mss.mss()
        self.templates = {}
        
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

    def click_human(self, template_key, threshold=0.75, speed=0.5):
        if template_key not in self.templates: return False
        
        frame = self.capture_screen()
        res = cv2.matchTemplate(frame, self.templates[template_key], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            h, w = self.templates[template_key].shape[:2]
            cx = max_loc[0] + w // 2
            cy = max_loc[1] + h // 2
            
            print(f"[NAVIGATOR] Ljudski potez na: {template_key}")
            pyautogui.moveTo(cx, cy, duration=speed, tween=pyautogui.easeOutQuad)
            pyautogui.click()
            return True
        return False

    def handle_popups(self):
        print("[NAVIGATOR] Provera pop-upova...")
        popups_clicked = 0
        
        start_wait = time.time()
        first_popup_found = False
        
        while time.time() - start_wait < 2.0:
            if self.click_human('btn_new_game_big', speed=0.3):
                first_popup_found = True
                popups_clicked += 1
                break
            if self.click_human('btn_new_game_small', speed=0.3):
                first_popup_found = True
                popups_clicked += 1
                break
            time.sleep(0.1)

        if first_popup_found:
            time.sleep(0.2)
            
            rapid_start = time.time()
            while time.time() - rapid_start < 2.0:
                if self.click_human('btn_confirm', speed=0.3):
                    print("[NAVIGATOR] Potvrda kliknuta!")
                    popups_clicked += 1
                    break
                # Provera i za stari mali ako se pojavi kao drugi
                if self.click_human('btn_new_game_small', speed=0.3):
                    popups_clicked += 1
                    break

        return popups_clicked

    def wait_for_game_start(self, urgent_mode=False):
        mode_str = "URGENTNO" if urgent_mode else "OPUŠTENO"
        print(f"[NAVIGATOR] Čekam start igre...")
        
        timeout = 15
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            mouse_speed = 0.2 if urgent_mode else 0.5
            if self.click_human('click_to_start', speed=mouse_speed, threshold=0.7):
                continue

            frame = self.capture_screen()
            if 'ingame_check' in self.templates:
                res = cv2.matchTemplate(frame, self.templates['ingame_check'], cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                
                if max_val >= 0.7:
                    print("[NAVIGATOR] Igra je spremna! Predajem kontrolu.")
                    return True
            
            if urgent_mode:
                pass 
            else:
                time.sleep(0.5)

        return False

    def start_whack_a_zombie(self):
        if not self.click_human('main_menu_minigames', speed=0.6): return False
        
        time.sleep(1.0)
        
        if not self.click_human('icon_whack', speed=0.6): return False
        
        clicks_count = self.handle_popups()
        is_urgent = (clicks_count >= 2)
        return self.wait_for_game_start(urgent_mode=is_urgent)