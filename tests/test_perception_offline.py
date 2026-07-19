import sys, os
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
import cv2
import config
from bot.vision.capture import ScreenCapture
from bot.vision.hud_reader import HudReader
from bot.vision.template_match import find_template

screen = ScreenCapture()
screen.gx, screen.gy, screen.gw, screen.gh = 320, 0, 1920, 1440

frame = cv2.imread("assets_capture/frame_13.png")

# 1) karte: sun=300, sve tri treba SPREMNE
hud = HudReader(screen)
states = hud.card_states(frame)
print("slotovi (potato, buster, ice):", states)
assert states == (True, True, True), states

# remap kao u real_env
by_name = dict(zip(config.CARD_ORDER, states))
canonical = (by_name["gravebuster"], by_name["potato_mine"], by_name["ice"])
print("kanonski (buster, mina, led):", canonical)

# 2) HUD marker u uglu okvira igre
tpl = cv2.imread("assets/ingame_check.png")
corner = frame[0:int(0.25*1440), 320:320+int(0.20*1920)]
res = cv2.matchTemplate(corner, tpl, cv2.TM_CCOEFF_NORMED)
import numpy as np
print(f"HUD marker u uglu: {res.max():.3f} (prag 0.70)")
assert res.max() >= 0.70

# 3) skalirani opseg sunca
gs = 1440/600.0
print(f"opseg povrsine sunca: {config.SUN_AREA_MIN_NATIVE*gs**2:.0f} - {config.SUN_AREA_MAX_NATIVE*gs**2:.0f} px^2 (staro: 1700-8000)")
print("SVE OK")
