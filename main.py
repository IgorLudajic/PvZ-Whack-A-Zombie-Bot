import time
import pyautogui
from navigator import PvZNavigator
from gameplay import WhackAZombieBot

def main():
    print("=========================================")
    print("   PvZ WHACK-A-ZOMBIE BOT v1.0")
    print("=========================================")
    
    pyautogui.FAILSAFE = True

    try:
        nav = PvZNavigator()
        
        print("\n--> Molim te otvori igru (Main Menu) i ne diraj miš.")
        print("--> Počinjem za 3 sekunde...")
        time.sleep(3)

        if nav.start_whack_a_zombie():
            print("\n[INFO] Igra je pronađena i pokrenuta.")
            print("[INFO] Učitavam agenta za borbu...")
            
            time.sleep(2) 
            
            bot = WhackAZombieBot()
            bot.run()
            
        else:
            print("\n[ERROR] Navigacija nije uspela. Proveri da li je igra u fokusu.")

    except FileNotFoundError as e:
        print(f"\n[CRITICAL] FALE FAJLOVI: {e}")
        print("Proveri da li su sve slike u 'assets' folderu.")
    except Exception as e:
        print(f"\n[CRITICAL] Neočekivana greška: {e}")

if __name__ == "__main__":
    main()