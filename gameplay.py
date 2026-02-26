import cv2
import numpy as np
import pyautogui
import mss
import time
import random
import ctypes
import math
import os
import sys
import csv
from datetime import datetime
from ultralytics import YOLO

try:
    ctypes.windll.user32.SetProcessDPIAware()
except AttributeError:
    pass

LOWER_YELLOW = np.array([16, 52, 172])
UPPER_YELLOW = np.array([40, 255, 255])

class GameStatsLogger:
    def __init__(self, filename="game_stats.csv"):
        self.filename = filename
        self.ensure_file_exists()

    def ensure_file_exists(self):
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Result", "Duration (s)", "Zombies Killed"])

    def log_game(self, result, duration, kills):
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result, f"{duration:.2f}", kills])
        self.print_win_rate()

    def print_win_rate(self):
        wins = 0
        total = 0
        with open(self.filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                total += 1
                if row["Result"] == "WIN":
                    wins += 1
        if total > 0:
            win_rate = (wins / total) * 100
            print(f"\n[STATISTICS] Odigrano: {total} | Pobede: {wins} | Win Rate: {win_rate:.2f}%")
        else:
            print("\n[STATISTICS] Nema zabeleženih partija.")

class WhackAZombieBot:
    def __init__(self):
        self.sct = mss.mss()
        self.logger = GameStatsLogger()
        self.start_time = time.time()
        self.zombies_killed_session = 0
        
        self.show_debug = True
        self.debug_window_name = "Bot AI Vision"
        
        self.debug_w = 640
        self.debug_h = 480
        
        self.last_screenshot_time = time.time()

        print("\n[INIT] Učitavam model i slike...")
        try:
            self.model = YOLO('best.pt')
        except:
            print("[CRITICAL] Nema best.pt!")
            exit()

        self.moneybag_img = None
        if os.path.exists('assets/moneybag.png'):
            img = cv2.imread('assets/moneybag.png')
            self.moneybag_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.gameover_img = None
        if os.path.exists('assets/gameover.png'):
            img = cv2.imread('assets/gameover.png')
            self.gameover_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.class_config = {
            'zombie':         {'clicks': 1, 'ignore': 0.15, 'color': (0, 255, 0)},    
            'conehead':       {'clicks': 2, 'ignore': 0.25, 'color': (0, 255, 255)},   
            'buckethead':     {'clicks': 3, 'ignore': 0.35, 'color': (0, 0, 255)},     
            'damaged_buckethead': {'clicks': 2, 'ignore': 0.25, 'color': (0, 100, 255)} 
        }
        
        self.ignored_targets = []
        pyautogui.PAUSE = 0.0
        self.screen_w, self.screen_h = pyautogui.size()

    def capture_screen_data(self):
        monitor = self.sct.monitors[1]
        img_bgra = np.array(self.sct.grab(monitor))
        img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        return img_rgb, img_hsv, img_bgr, monitor['left'], monitor['top'], monitor['width'], monitor['height']

    def clamp_val(self, val, min_v, max_v):
        return int(max(min_v, min(val, max_v)))

    def safe_move_and_click(self, x, y, clicks=1, duration=0.1, monitor_bbox=None, hold_time=0.04):
        safe_x = self.clamp_val(x, 0, self.screen_w - 1)
        safe_y = self.clamp_val(y, 0, self.screen_h - 1)

        if monitor_bbox:
            l, t, w, h = monitor_bbox
            safe_x = self.clamp_val(safe_x, l + 5, l + w - 5)
            safe_y = self.clamp_val(safe_y, t + 5, t + h - 5)

        if self.show_debug:
            if safe_x < self.debug_w and safe_y < self.debug_h:
                return

        pyautogui.moveTo(safe_x, safe_y, duration=duration, _pause=False)
        
        for i in range(clicks):
            pyautogui.mouseDown(_pause=False)
            time.sleep(hold_time)
            pyautogui.mouseUp(_pause=False)
            
            if clicks > 1 and i < clicks - 1:
                time.sleep(0.05)

    def execute_grave_buster_sequence(self, card_x, card_y, grave_x, grave_y, bbox):
        l, t, w, h = bbox
        target_grave_x = self.clamp_val(grave_x, l+20, l+w-20)
        target_grave_y = self.clamp_val(grave_y - 20, t+80, t+h-20)

        #print(f"[BOT] Sadim Gravebuster...")
        self.safe_move_and_click(card_x, card_y, clicks=1, duration=0.15, monitor_bbox=bbox, hold_time=0.08)
        time.sleep(0.1)
        self.safe_move_and_click(target_grave_x, target_grave_y, clicks=0, duration=0.2, monitor_bbox=bbox)
        pyautogui.click(x=target_grave_x, y=target_grave_y, duration=0.05)
        time.sleep(0.5)

    def pro_gamer_move(self, start_x, start_y, target_x, target_y, clicks=1, is_collecting=False, bbox=None, zombie_count=0):
        if is_collecting:
            duration = random.uniform(0.05, 0.09)
            hold = 0.03
        else:
            if zombie_count <= 3:
                duration = random.uniform(0.12, 0.18) 
                hold = 0.05
            elif zombie_count <= 8:
                duration = random.uniform(0.05, 0.08)
                hold = 0.025
            else:
                duration = random.uniform(0.01, 0.03)
                hold = 0.01

        self.safe_move_and_click(target_x, target_y, clicks=clicks, duration=duration, monitor_bbox=bbox, hold_time=hold)

    def is_ignored(self, x, y, radius=30):
        current_time = time.time()
        self.ignored_targets = [t for t in self.ignored_targets if t[2] > current_time]
        
        if x < self.debug_w and y < self.debug_h:
            return True

        for (ix, iy, _) in self.ignored_targets:
            if abs(x - ix) < radius and abs(y - iy) < radius:
                return True
        return False

    def add_to_ignore(self, x, y, duration=1.0):
        self.ignored_targets.append((x, y, time.time() + duration))

    def find_suns_hsv(self, img_hsv, offset_left, offset_top, debug_img=None):
        mask = cv2.inRange(img_hsv, LOWER_YELLOW, UPPER_YELLOW)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sun_targets = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 1700 < area < 8000:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if cy < 100: continue
                    abs_x = cx + offset_left
                    abs_y = cy + offset_top
                    
                    if not self.is_ignored(abs_x, abs_y, radius=40):
                        sun_targets.append({'x': abs_x, 'y': abs_y, 'clicks': 1, 'type': 'sun', 'ignore': 0.6})
                        
                        if debug_img is not None:
                             cv2.circle(debug_img, (cx, cy), 20, (0, 255, 255), 2)
                             
        return sun_targets

    def sort_by_proximity(self, targets, start_x, start_y):
        if not targets: return []
        sorted_targets = []
        curr_x, curr_y = start_x, start_y
        remaining = targets.copy()
        while remaining:
            closest_idx = -1
            min_dist = float('inf')
            for i, t in enumerate(remaining):
                dist = math.hypot(t['x'] - curr_x, t['y'] - curr_y)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            best_target = remaining.pop(closest_idx)
            sorted_targets.append(best_target)
            curr_x, curr_y = best_target['x'], best_target['y']
        return sorted_targets

    def run(self):
        print("[GAMEPLAY] REŽIM: VISUAL OVERLAY + SAFETY LOCK + DYNAMIC LEAD.")
        self.start_time = time.time()

        if self.show_debug:
            cv2.namedWindow(self.debug_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.debug_window_name, self.debug_w, self.debug_h)
            cv2.moveWindow(self.debug_window_name, 0, 0) 
        
        while True:
            frame_rgb, frame_hsv, frame_bgr, off_l, off_t, mon_w, mon_h = self.capture_screen_data()
            debug_frame = frame_bgr.copy()

            mon_bbox = (off_l, off_t, mon_w, mon_h)
            curr_x, curr_y = pyautogui.position()
            current_duration = time.time() - self.start_time

            if current_duration > 260:
                self.logger.log_game("WIN", current_duration, self.zombies_killed_session)
                print(f"\n[TIMER] Isteklo 260 sekundi! Upisujem POBEDU i gasim.")
                time.sleep(1)
                sys.exit()

            danger_limit_x = off_l + (mon_w * 0.35)

            if self.show_debug:
                overlay = debug_frame.copy()
                cv2.rectangle(overlay, (0, 0), (self.debug_w, self.debug_h), (50, 50, 50), -1)
                cv2.addWeighted(overlay, 0.3, debug_frame, 0.7, 0, debug_frame)
                cv2.putText(debug_frame, "BOT SAFE ZONE (NO CLICK)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if self.moneybag_img is not None:
                res = cv2.matchTemplate(frame_rgb, self.moneybag_img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val >= 0.8:
                    h, w = self.moneybag_img.shape[:2]
                    cx, cy = max_loc[0] + w//2, max_loc[1] + h//2
                    self.pro_gamer_move(curr_x, curr_y, cx + off_l, cy + off_t, clicks=1, is_collecting=True, bbox=mon_bbox)
                    duration = time.time() - self.start_time
                    self.logger.log_game("WIN", duration, self.zombies_killed_session)
                    print("\n[STATISTICS] POBEDA UPISANA!")
                    time.sleep(2)
                    sys.exit()

            if self.gameover_img is not None:
                res = cv2.matchTemplate(frame_rgb, self.gameover_img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val >= 0.8:
                    duration = time.time() - self.start_time
                    self.logger.log_game("LOSS", duration, self.zombies_killed_session)
                    print("\n[STATISTICS] PORAZ UPISAN!")
                    sys.exit()

            results = self.model.predict(source=frame_rgb, conf=0.15, imgsz=1280, verbose=False)
            result = results[0]

            combat_queue = []
            gravebuster_candidates = []
            detected_graves = []

            for box in result.boxes:
                cls_name = result.names[int(box.cls[0])]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                abs_x, abs_y = cx + off_l, cy + off_t

                if self.show_debug:
                    box_color = (255, 0, 255)
                    if cls_name in self.class_config:
                        box_color = self.class_config[cls_name]['color']
                    cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                    cv2.putText(debug_frame, cls_name, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                if cls_name == 'gravebuster':
                    gravebuster_candidates.append((abs_x, abs_y))
                elif cls_name == 'grave':
                    if not self.is_ignored(abs_x, abs_y, radius=40):
                        detected_graves.append({'x': abs_x, 'y': abs_y})
                elif cls_name in self.class_config:
                    aim_y = abs_y + 50
                    if not self.is_ignored(abs_x, aim_y):
                        cfg = self.class_config[cls_name]
                        combat_queue.append({
                            'x': abs_x, 'y': aim_y,
                            'clicks': cfg['clicks'],
                            'ignore': cfg['ignore'],
                            'type': cls_name
                        })
                    else:
                         if self.show_debug:
                             cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (100, 100, 100), 2)
            
            current_loop_time = time.time()
            if current_loop_time - self.last_screenshot_time >= 78: 
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"screenshot_{timestamp}.png"
                cv2.putText(debug_frame, f"AUTO-SHOT: {timestamp}", (10, self.debug_h - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imwrite(filename, debug_frame)
                print(f"[INFO] Screenshot sačuvan: {filename}")
                self.last_screenshot_time = current_loop_time

            active_zombies_count = len(combat_queue)
            
            if combat_queue:
                combat_queue.sort(key=lambda z: z['x'])

            most_dangerous_zombie = combat_queue[0] if combat_queue else None
            is_critical_danger = most_dangerous_zombie and most_dangerous_zombie['x'] < danger_limit_x
            
            if self.show_debug:
                cv2.line(debug_frame, (int(mon_w * 0.35), 0), (int(mon_w * 0.35), mon_h), (0, 0, 255), 2)

            prediction_offset = 0
            if active_zombies_count > 12: prediction_offset = 65 
            elif active_zombies_count > 7: prediction_offset = 40
            elif active_zombies_count > 3: prediction_offset = 15 

            if is_critical_danger:
                target = most_dangerous_zombie
                aim_x = target['x'] - prediction_offset
                
                if self.show_debug:
                    cv2.line(debug_frame, (int(curr_x - off_l), int(curr_y - off_t)), (int(aim_x - off_l), int(target['y'] - off_t)), (0, 0, 255), 3)

                self.add_to_ignore(target['x'], target['y'], duration=target['ignore'])
                self.pro_gamer_move(curr_x, curr_y, aim_x, target['y'],
                                    clicks=target['clicks'], bbox=mon_bbox,
                                    zombie_count=active_zombies_count)
                self.zombies_killed_session += 1
                
                if self.show_debug:
                    cv2.imshow(self.debug_window_name, debug_frame)
                    cv2.waitKey(1)
                continue

            if active_zombies_count <= 8:
                suns = self.find_suns_hsv(frame_hsv, off_l, off_t, debug_img=debug_frame if self.show_debug else None)
                if suns:
                    sorted_suns = self.sort_by_proximity(suns, curr_x, curr_y)
                    for sun in sorted_suns:
                        if not self.is_ignored(sun['x'], sun['y']):
                            if self.show_debug:
                                 cv2.line(debug_frame, (int(curr_x - off_l), int(curr_y - off_t)), (int(sun['x'] - off_l), int(sun['y'] - off_t)), (0, 255, 255), 2)
                                 cv2.imshow(self.debug_window_name, debug_frame)
                                 cv2.waitKey(1)

                            self.pro_gamer_move(curr_x, curr_y, sun['x'], sun['y'],
                                                clicks=1, is_collecting=True, bbox=mon_bbox,
                                                zombie_count=active_zombies_count)
                            self.add_to_ignore(sun['x'], sun['y'], duration=sun['ignore'])
                            curr_x, curr_y = sun['x'], sun['y']
            
            if combat_queue:
                if active_zombies_count > 8:
                    batch_size = 4
                    batch_wait_time = 0.0
                elif active_zombies_count > 4:
                    batch_size = 2
                    batch_wait_time = 0.02
                else:
                    batch_size = 1
                    batch_wait_time = 0.05

                targets_to_hit = combat_queue[:batch_size]
                
                if active_zombies_count <= 8:
                    targets_to_hit = self.sort_by_proximity(targets_to_hit, curr_x, curr_y)

                for target in targets_to_hit:
                    aim_x = target['x'] - prediction_offset
                    
                    if self.show_debug:
                        cv2.line(debug_frame, (int(curr_x - off_l), int(curr_y - off_t)), (int(aim_x - off_l), int(target['y'] - off_t)), (0, 255, 0), 2)
                        cv2.imshow(self.debug_window_name, debug_frame)
                        cv2.waitKey(1)

                    self.add_to_ignore(target['x'], target['y'], duration=target['ignore'])
                    self.pro_gamer_move(curr_x, curr_y, aim_x, target['y'],
                                        clicks=target['clicks'], bbox=mon_bbox,
                                        zombie_count=active_zombies_count)
                    
                    curr_x, curr_y = aim_x, target['y']
                    self.zombies_killed_session += 1
                    
                    if batch_size > 1:
                        time.sleep(batch_wait_time)

                if active_zombies_count <= 2 and gravebuster_candidates and detected_graves:
                      gravebuster_candidates.sort(key=lambda coord: coord[1])
                      detected_graves.sort(key=lambda g: g['x'])
                      self.execute_grave_buster_sequence(
                        gravebuster_candidates[0][0], gravebuster_candidates[0][1],
                        detected_graves[0]['x'], detected_graves[0]['y'],
                        mon_bbox
                      )
                      self.add_to_ignore(detected_graves[0]['x'], detected_graves[0]['y'], duration=4.0)
                
                if self.show_debug:
                    cv2.imshow(self.debug_window_name, debug_frame)
                    cv2.waitKey(1)
                continue

            if self.show_debug:
                cv2.imshow(self.debug_window_name, debug_frame)

            if cv2.waitKey(1) == ord('q'):
                break
