# fred_train.py
import sys
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

from tkinter import simpledialog

from datetime import datetime
import matplotlib

matplotlib.use("TkAgg")

import config
import matplotlib.pyplot as plt
import csv
from datetime import datetime

import time
import pyautogui
import mss

ausgabedatei = "mnist_dataset/fredy_train.csv"
if len(sys.argv) > 1:
    h1="mnist_dataset/"
    h2=sys.argv[1]
    h3=".csv"
    ausgabedatei = h1+h2+h3

print("CSV Ausgabe:", ausgabedatei)


inputfile=""



letztes_speichern = 0
speicher_intervall = 0.1    # Sekunden

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ # Global
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ # Frame
print(f"-Start-------------------------------------------------------------------------------------------------")

print(f"----------------------cammax2.py-by-Elvis-Dizdarevic---------------------------------------------------")

print(f"-------------------------------------------------------------------------------------------------------")
print(f"Steuerung: + und - Resize Frame.")
print(f"Steuerung: 1 durch Anzeigen schalten.")
print(f"Steuerung: 6 + 7 Trashold angleichen.")
print(f"Steuerung: 8 + 9 Schwellenwert für Counter.")
print(f"Steuerung: 0 Counter zurücksetzen.")
print(f"Steuerung: P Detector Graph fest Pinen am Bildschirm.")
print(f"Steuerung: Oben und Unten Helligkeit.")
print(f"Steuerung: Links und Rechts Sättigung.")

print(f"Steuerung: ESC oder Q sind Programm Beenden.")

print(f"Programm Start mit Parameterübergabe möglich: python3 cammax2.py [ausgabe_Counter.txt].")
print(f"Speichert den Zeitstempel in [ausgabe_Counter.txt]  wenn Counter zählt.")
print(f"Default ist [Counter_log.txt].")

print(f"Anwendung: mit P einen Bildschirm auswählen. Bildschirm anpassen, weil nur Weisse Punkte werden gezählt.")
print(f"-------------------------------------------------------------------------------------------------------")

print(f"herodacles@gmail.de 2025")
print(f"------------------------")
print(f"-------------------------------------------------------------------------------------------------------")






ziel = np.zeros((130,240,3), dtype=np.uint8)
ziel1 = np.zeros((130,240), dtype=np.uint8)
print(ziel1.shape)
print(ziel.shape)
# Seitenverhältnis 16:9 (~1.833)
ASPECT_RATIO = 1980 / 1080

# HD-Auflösung initial
# HD_WIDTH dynamisch ändern, HD_HEIGHT automatisch berechnen
HD_WIDTH = 240
HD_HEIGHT = int(HD_WIDTH / ASPECT_RATIO)

MAX_WIDTH, MAX_HEIGHT = 10000,int(10000 / ASPECT_RATIO) # 1,833 
MIN_WIDTH, MIN_HEIGHT = 240, int(300 / ASPECT_RATIO)   # Minimalgröße

End_HD_WIDTH, End_HD_HEIGHT = 240, 130

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ # Video
USE_DESKTOP = False
MONITOR_INDEX = 0
DESKTOP_INDEX = 0
DESKTOP_ANZAHL = 1
# Desktop / Arbeitsflächen
ARBEITSFLAECHE = 0
ANZAHL_ARBEITSFLAECHEN = 4

USE_IMAGE_FILE = False
IMAGE_PATH = ""
image_frame = None


USE_VIDEO_FILE = False   # True = Video, False = Webcam
VIDEO_PATH = "/home/herodacles/Videos/test.MOV"
CAM_INDEX = 0

OUTPUT_PATH = "output_record.mp4"  # Ausgabedatei
FPS = 30  # Frames per Second
WIDTH, HEIGHT = 1980,1080  # Auflösung der Aufnahme
# VideoWriter für Aufnahme einrichten



fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # AVI mit XVID-Codec
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))
if not out.isOpened():
    print("VideoWriter konnte nicht geöffnet werden")
else:
    print("VideoWriter OK")
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ # Kippschalter

kippschalterf1=10; # welche anzeige startet programm

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

prev_frame = None

my_thresh_thresh = 10 # 0-255 alles unter trasch schwarz rest weiss
my_thresh_maxval = 255 # 0-255 wert wo aufgefüllt wird
my_thresh_type = 1 # gedacht für kippschalter für tüpumschaltung modi möglich

CANNY_MIN = 50
CANNY_MAX = 150
DILATE_ITER = 2          # für dickere Kanten
# === Kernel für Dilate/Erode ===
kernel = np.ones((3, 3), np.uint8)

# Globale Einstellungen
brightness_step = 5   # Schrittweite Helligkeit
saturation_step = 5   # Schrittweite Sättigung
v_offset = 0
s_scale = 1.0

#---------------------------------------------------- # für bildauschnitt

#---------------------------------------------------- # für daten 

MAX_TOTAL_DATA = 50000       # Gesamtanzahl gespeicherte Werte
WINDOW_WIDTH_DATA = 800           # Breite des Anzeigefensters
HEIGHT_DATA= 250

# Beispiel-Daten
data = np.zeros((MAX_TOTAL_DATA, 2), dtype=np.int32) #anlegen

for i in range(MAX_TOTAL_DATA):                      # füllen
    data[i, 0] = np.random.randint(0, 3)
    data[i, 1] = np.random.randint(0, 3)

# Index des neuesten Wertes
index_DATA = WINDOW_WIDTH_DATA

# Bild erstellen
img_DATA = np.zeros((HEIGHT_DATA, WINDOW_WIDTH_DATA, 3), dtype=np.uint8)

# für weiterverarbeitung
schwellwert=30 #

# für zählen
ergebnis_index=0
ergebnis = np.zeros(MAX_TOTAL_DATA, dtype=bool)  #ergebnis 
ergebnis_action = False

aufbereitung_index=0
aufbereitung = np.zeros(MAX_TOTAL_DATA, dtype=bool)  #ergebnis 
aufbereitung_action = False

counter = 0
counter_flag=False

DATEI = "Counter_log.txt"
if len(sys.argv) > 1:
    DATEI = sys.argv[1]


with open(DATEI, "w", encoding="utf-8"):
    pass  # Datei wird geöffnet und überschriben und sofort wieder geschlossen, bleibt leer




def save_ziel1_csv(ziel1, dateiname=ausgabedatei,label=0):
    """
    Speichert ein Detector-Bild (Array) als CSV.
    
    ziel1:
        2D numpy Array, z.B. (130,240)
    """

    # Sicherheit: auf 2D bringen
    #if len(ziel1.shape) == 3:
        #ziel1 = ziel1[:, :, 0]
    
    #print("ziel1 Form:", ziel1.shape)
    #print(ziel1[0,0:10])
    
    #print("VOR CSV:")
    #print(ziel1.shape)
    #print(ziel1[0,0:50])
    #print("MIN:", ziel1.min())
    #print("MAX:", ziel1.max())
    
    ys, xs = np.where(ziel1 > 0)
    '''
    if len(xs) > 0:
       print("Objekt Bereich:")
       print("X:", xs.min(), xs.max())
       print("Y:", ys.min(), ys.max())
    else:
       print("Kein Inhalt")  
    '''

    #cv2.imshow("VOR CSV", ziel1) 
    
    bild = ziel1.flatten()
    
    #print("Speichern Pixel:", len(bild))
    #print("Form:", ziel1.shape)
    
    
    with open(dateiname, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Zeitstempel als erste Zeile
        
       
        writer.writerow(
            [label] + bild.tolist()
        )



def csv_status(dateiname=ausgabedatei):
    if os.path.exists(dateiname):
        groesse = os.path.getsize(dateiname)
        mb = groesse / (1024 * 1024)

        with open(dateiname, "r", encoding="utf-8") as f:
            zeilen = sum(1 for _ in f)

        print(f"CSV: {ausgabedatei}")
        print(f"Größe: {mb:.2f} MB")
        print(f"Bilder gespeichert: {zeilen}")
    else:
        print("CSV existiert noch nicht")

def label_auswahl():
    root = tk.Tk()
    root.title("Label auswählen")
    root.geometry("300x200")

    klassen = {
        "Menschen":0,
        "Tiere":1,
        "Pflanzen":2,
        "Pilze":3,
        "Bakterien":4,
        "Einzeller":5,
        "Insekten":6,
        "Fische":7,
        "Vögel":8,
        "Säugetiere":9,
        "Reptilien":10,
        "Amphibien":11,
        "Gesteine":12,
        "Mineralien":13,
        "Metalle":14,
        "Wasser":15,
        "Wolken/Wetter":16,
        "Landschaften":17,
        "Gebäude":18,
        "Fahrzeuge":19
    }

    auswahl_text = tk.StringVar(root)
    auswahl_text.set("Menschen")

    box = tk.OptionMenu(
        root,
        auswahl_text,
        *klassen.keys()
    )
    box.pack(pady=20)

    ergebnis = {"wert":0}

    def uebernehmen():
        ergebnis["wert"] = klassen[auswahl_text.get()]
        root.quit()

    
    button = tk.Button(
        root,
        text="OK",
        command=uebernehmen
    )
    button.pack()

    root.mainloop()
    root.destroy()

    return ergebnis["wert"]





#label = label_auswahl()
label = 0
print("Label:", label)
 #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
                                                                             
                                                                             
                                                                             # Für Video
def choose_file():
    root = tk.Tk()
    root.withdraw()  # kein extra Fenster
    file_path = filedialog.askopenfilename(
        title="Datei auswählen",
        filetypes=[
            ("Alle Dateien", "*.*"),
            ("Video-Dateien", "*.mp4 *.avi *.mkv *.mov *.webm")
            
        ]
    )
    root.destroy()
    return file_path

def choose_file2():
    root = tk.Tk()
    root.withdraw()  # kein extra Fenster
    file_path = filedialog.askopenfilename(
        initialdir="/home/herodacles/Videos",
        title="Datei auswählen",
        filetypes=[
            ("Alle Dateien", "*.*"),
            ("Video-Dateien", "*.mp4 *.avi *.mkv *.mov *.webm")
            
        ]
    )
    root.destroy()
    return file_path


def desktop_anzeigen(index):

    from screeninfo import get_monitors

    monitore = get_monitors()

    #print("Monitore:", len(monitore))

    if index >= len(monitore):
        index = 0

    monitor = monitore[index]
    
    '''
    print(
        "Desktop gewählt:",
        index,
        monitor.width,
        "x",
        monitor.height
    )
    '''
    return monitor

def arbeitsflaeche_wechseln(index):

    import subprocess

    subprocess.run(
        [
            "wmctrl",
            "-s",
            str(index)
        ]
    )
'''
    print(
        "Arbeitsfläche:",
        index + 1
    )
'''




################
###############

if USE_VIDEO_FILE:
    VIDEO_PATH = choose_file()
   





 #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
x1, y1, x2, y2 = 0, 0, 0, 0
speichern_pin=False
drawing_pin=False
drawing = True
roi_ready = False
ROI_SIZE_X = config.input_x
ROI_SIZE_Y = config.input_y
 
                                                                               # Mouse
# ---------- Maus Callback ----------
def mouse_callback(event, x, y, flags, param):
     
     if drawing_pin==True:
         global x1, y1, x2, y2, drawing, roi_ready
         if event == cv2.EVENT_LBUTTONDOWN:
             scalex =HD_WIDTH / End_HD_WIDTH    #skaliern auf von aussen nach innen
             scaley = HD_HEIGHT /End_HD_HEIGHT
             x =   (x * scalex)
             y =   (y * scaley)
             x =int(x)
             y=int(y)
             x1 = x - ROI_SIZE_X // 2
             y1 = y - ROI_SIZE_Y // 2            
             x2 = x + ROI_SIZE_X // 2
             y2 = y + ROI_SIZE_Y // 2
             roi_ready = True
             
             #print(x1)
             #print(y1)
    #--------------------------------------------------
 #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                                                                             # Cam's
# Kamera öffnen
def find_cameras(max_test=4):
    cams = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cams.append(i)
        cap.release()
    return cams


cameras = find_cameras()

if not cameras:
    print("❌ Keine Webcam gefunden")
    exit()

cam_index = 0


if USE_VIDEO_FILE:
    cap = cv2.VideoCapture(VIDEO_PATH)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    if FPS <= 0:
       FPS = 30  # Fallback

    print("🎞 Video-Datei als Input")
else:
    cap = cv2.VideoCapture(cameras[cam_index], cv2.CAP_V4L2)
    print("🎥 Aktive Kamera:", cameras[cam_index])


if not cap.isOpened():
    print("Kamera konnte nicht geöffnet werden")
    exit()

 #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                                                                             # Haupt While

while True:

    if USE_DESKTOP:

        import pyautogui

        monitor = desktop_anzeigen(
            MONITOR_INDEX
        )
        
        bildschirm = pyautogui.screenshot(
            region=(
                monitor.x,
                monitor.y,
                monitor.width,
                monitor.height
            )
        )
        frame = np.array(bildschirm)

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        ret = True

    else:

        if USE_IMAGE_FILE:
    
            frame = image_frame.copy()
            ret = True
    
        else:
    
            ret, frame = cap.read()   
        
        
    if not ret:  # Ende des Videos erreicht
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Zurück zum ersten Frame
        continue
    color = cv2.resize(frame, (HD_WIDTH, HD_HEIGHT))                                                   #frame holen jetzt color
    color_bg = color #ohne rausch übergeben
    
    # --- Farben blasser machen ---
    hsv = cv2.cvtColor(color_bg, cv2.COLOR_BGR2HSV)                                               # 3 kanal signal holen
    h, s, v = cv2.split(hsv)                                                                      # signal auf variablen aufsplitten
    # Helligkeit und Sättigung anpassen                               
    v = np.clip(v + v_offset, 0, 255)                                                               
    s = np.clip((s * s_scale).astype(np.uint8), 0, 255)
    color = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    
    # --- dunkle Farben entfernen ---
    h, s, v = cv2.split(hsv)
    v[v < 1] = 255   # alles Dunkle wird weiß
    color_bg = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    hsv = cv2.cvtColor(color_bg, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
   
    # --- helle Farben abdunkeln ---
    v[v > 250] = 250  # Helligkeit stark reduzieren, Farbtöne bleiben
    color_bg = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    
    
    # 2. Kanten extrahieren
    
    gray = cv2.cvtColor(color_bg, cv2.COLOR_BGR2GRAY)                # holt sich- color -eingestelten rahmen + hell + sätiigung 
    #gray = cv2.medianBlur(gray, 3)    
    #gray = cv2.GaussianBlur(gray, (3,3 ), 0)                      # gausssian blur
    
    # kanten fertig extrahiert
    edges = cv2.Canny(gray, 50,150)
    #edges = cv2.Canny(gray, my_thresh_thresh,my_thresh_maxval)
    
    #edges = cv2.dilate(edges, kernel, iterations=DILATE_ITER)
    #edges = cv2.bitwise_not(edges)
	
	#_, mask = cv2.threshold(edges, my_thresh_thresh, my_thresh_maxval, cv2.THRESH_BINARY)
    _, mask = cv2.threshold(edges, my_thresh_thresh, 150, cv2.THRESH_BINARY)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	
    #kanten in frame schreiben
    output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)               # hier sind kanten fertig - output
    
    #Bewegungsdetektion
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame is not None else gray.copy()   # mit fehlerabfang richtig # frame -prev_frame -zu 1 kanal Graustufen 

    # 2. Größe angleichen
    #gray_frame = cv2.resize(gray_frame, (HD_WIDTH, HD_HEIGHT))
    #prev_gray = cv2.resize(prev_gray, (HD_WIDTH, HD_HEIGHT))

    # 3. Differenz berechnen
    #gray_frame = cv2.resize(gray_frame, (HD_WIDTH, HD_HEIGHT)) 
    prev_gray = cv2.resize(prev_gray, (HD_WIDTH, HD_HEIGHT))     
    diff = cv2.absdiff(gray, prev_gray)
 
    # 4. Signifikante Bewegung maskieren
    #_, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)  # standart
    #_, mask = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)  # weniger durchlassen bei bewegung
    #_, mask = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)  # mehr durchlassen bei bewegung fast dulässiges flimmer zu fill dauerhaft sichtbar viel
    
    my_thresh_type = my_thresh_type
    _, mask = cv2.threshold(diff, my_thresh_thresh, my_thresh_maxval, cv2.THRESH_BINARY)  # mehr durchlassen bei bewegung fast dulässiges flimmer zu fill dauerhaft sichtbar viel
	

    # 5. Bewegung sichtbar machen, Rest schwarz
    diff_frame = cv2.merge([mask, mask, mask])  # 3-Kanal-Bild für Anzeige

    # 6. Vorheriges Frame speichern
    prev_frame = color.copy()


  
  
  
  
    # 3. Farbe kombinieren
    
    mycolx=int(HD_WIDTH / 10) # variabel
    
    # --- Farb-Hintergrund (blockig) ---
    
     
    color_bg = cv2.resize(color, (mycolx,int(mycolx / ASPECT_RATIO)), interpolation=cv2.INTER_NEAREST)  #INTER_NEAREST,INTER_LINEAR
    
    

    noise = np.random.randint(0, 1, (64, 48, 3), dtype=np.int16)
    noise = cv2.resize(noise, (mycolx,int(mycolx / ASPECT_RATIO)), interpolation=cv2.INTER_NEAREST)
    
    color_bg = np.clip(color_bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # --- Farben blasser machen ---
    hsv = cv2.cvtColor(color_bg, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Helligkeit und Sättigung anpassen
    v = np.clip(v + v_offset, 0, 255)
    s = np.clip((s * s_scale).astype(np.uint8), 0, 255)
    # --- dunkle Farben entfernen ---
    h, s, v = cv2.split(hsv)
    #v[v < 25] = 255   # alles Dunkle wird weiß
    color_bg = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    hsv = cv2.cvtColor(color_bg, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # --- helle Farben abdunkeln ---
    #v[v > 225] = 220  # Helligkeit stark reduzieren, Farbtöne bleiben
    color_bg = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    cartoon_color = cv2.medianBlur(color_bg, 3)
    #cartoon_color = cv2.GaussianBlur(cartoon_color, (9, 9), 0)    
    
    
    
    
    #für zusammen führen alles auf gleiche grösse
    
    cartoon_color = cv2.resize(cartoon_color, (HD_WIDTH, HD_HEIGHT))
    output = cv2.resize(output, (HD_WIDTH, HD_HEIGHT))
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   																				# Kippschalter modi
    if kippschalterf1 > 10:
        kippschalterf1=1;
        print(f"{ kippschalterf1 } Anzeige ")
        
        #hier begient die letzte modelierung für die verschieden kipschalter anzeigen mit 1 durchwechsel
        
        
    if kippschalterf1 == 1:                                             #canny = output , cartoon_color = blockierte Farben                
        cartoon_color = cv2.bitwise_and(color, output)          #Rahmen canny + blockiertet farben gerändert
    if kippschalterf1 == 2:    
        cartoon_color = cv2.bitwise_or(cartoon_color, output)           #Rahmen canny + blockierte farben ausgeschniten -- macht weiss mit bunten rahmen
    if kippschalterf1 == 3:    
        cartoon_color = cv2.bitwise_not(output)                         #Rahmen canny schwarz hintergrund weiss 
    if kippschalterf1 == 4:    
        cartoon_color=output                                            #rahmen canny weiss hintergrund schwarz                                              
    if kippschalterf1 == 5:    
        cartoon_color=diff_frame                                             # zuweiszung erster motion frame
    if kippschalterf1 == 6:    
        cartoon_color = cv2.bitwise_and(color, diff_frame, mask=mask)             # zuweiszung and color +
    if kippschalterf1 == 7:    
        cartoon_color = cv2.bitwise_and(cartoon_color, diff_frame, mask=mask)             # zuweiszung and
    if kippschalterf1 == 8:    
        cartoon_color = cv2.bitwise_or( cv2.bitwise_not(output) , diff_frame, mask=mask)       # zuweiszung  or
    if kippschalterf1 == 9:    
        if prev_frame is None:                                               #hier Bewegungserkennung
            diff = cv2.absdiff(cartoon_color, outout)                        #frame übergabe einmalig      
            _, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY) 
            mask = cv2.dilate(edges, kernel, iterations=DILATE_ITER)                #maske
            diff_frame = cv2.bitwise_not(output, cartoon_color, mask)               #
        cartoon_color = cv2.bitwise_not( output , diff_frame , mask=mask)       # zuweiszung not
                                         
    if kippschalterf1 == 10:    
            
        #cartoon_color = cv2.bitwise_and(  gray,   mask)  
                           # zuweiszung not
        
        #cartoon_color = cv2.bitwise_and(  gray,   mask)                      # zuweiszung not
        cartoon_color = gray                           


	#nachdem maske ausgewählt ist geht weiter bild extrahiern




    paint = cartoon_color.copy()                               # zum zeichnen bild extrahiern
    
                                                               # zielimage anlegn
    if not roi_ready:
       ziel = np.zeros((config.input_x,config.input_y,3), dtype=np.uint8)
    

    # Rechteck anzeigen nur während Zeichnen oder wenn ROI fertig
    if roi_ready or drawing:
        # Rechteck zeichnen
        ratio= ROI_SIZE_X / ROI_SIZE_Y
        hy1=y1
        hx1=x1
        cv2.rectangle(cartoon_color, (hx1, hy1), (hx1+ROI_SIZE_X, hy1+ROI_SIZE_Y), (0,255,0), 2)
        h, w = paint.shape[:2]
        x_start = max(0, hx1)
        y_start = max(0, hy1)
        x_end = x_start + ROI_SIZE_X
        y_end = y_start + ROI_SIZE_Y
        
        ziel = paint[y_start:y_end, x_start:x_end]
        # Detector ROI auf 240x130 bringen
        ziel = cv2.resize(
            ziel,
            (ROI_SIZE_X, ROI_SIZE_Y),
            interpolation=cv2.INTER_AREA
        )
        
        #ziel = cv2.resize(ziel, (240,130))


        if len(ziel.shape) == 3:
            gray1 = cv2.cvtColor(ziel, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = ziel.copy()
                
       
#my_thresh_thresh = 10 # 0-255 alles unter trasch schwarz rest weiss
#my_thresh_maxval = 255 # 0-255 wert wo aufgefüllt wird
#my_thresh_type = 1 # gedacht für kippschalter für tüpumschaltung modi möglich
 
        
        edges1 = cv2.Canny(gray1, my_thresh_thresh, my_thresh_maxval)
        # spericherung
        ziel1 = gray1
        
        if speichern_pin:
            jetzt = time.time()
            
            if jetzt - letztes_speichern >= speicher_intervall:
                # Anzahl aktive Pixel im Bild prüfen
                weisse_pixel = cv2.countNonZero(ziel1)
                
                MIN_PIXEL = 20   # Mindestanzahl, anpassen
                
                if weisse_pixel >= MIN_PIXEL:
                    save_ziel1_csv(ziel1,label=label)
                    
                    letztes_speichern = jetzt
                    #print(f"Gespeichert: {weisse_pixel} Pixel")
                else:
                    print(f"Leer verworfen: {weisse_pixel} Pixel") 
            
            #print(ziel1)    
                
        # Graustufen
        if len(ziel.shape) == 3:
            gray_detector = cv2.cvtColor(ziel, cv2.COLOR_BGR2GRAY)
        else:
            gray_detector = ziel.copy()
        
        # Maske erzeugen
        _, detector_mask = cv2.threshold(
            gray_detector,
            schwellwert,
            255,
            cv2.THRESH_BINARY
        )
        
        ziel = cv2.resize(ziel, (240,130))
                
        
        if ziel.size > 0:
        
            if len(ziel.shape) == 3:
                gray = cv2.cvtColor(ziel, cv2.COLOR_BGR2GRAY)
            else:
                gray = ziel.copy()   
              
            mask1 = (gray == 0).astype("uint8")
            mask2 = (gray == 255).astype("uint8")
            
            anzahl_S = cv2.countNonZero(mask1)                        # werte ermitteln aus ausschnitt
            anzahl_W= cv2.countNonZero(mask2)

            cv2.putText(
                ziel, str(anzahl_W),
                (2,18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0,255,0),
                1
            )
            cv2.putText(
                ziel, str(anzahl_S),
                (2,38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0,0,255),
                1
            )
                      
                      
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++# detector grapf                                     Logik
   

#img_DATA
               # werte in daten schreiben grap anlegen frün
            if index_DATA < MAX_TOTAL_DATA:
               data[index_DATA, 0] = anzahl_W    #integer
               data[index_DATA, 1] = anzahl_S
               if data[index_DATA, 0] > schwellwert:                             #schwellenwert prüfen   
                   ergebnis[index_DATA]=True 
                   aufbereitung[index_DATA]=True 
               else:
                   ergebnis[index_DATA]=False
                   aufbereitung[index_DATA]=False 
               
            else:
               index_DATA = WINDOW_WIDTH_DATA ;
            
            
            if aufbereitung [index_DATA-3]: #  rot füllen
                if aufbereitung [ index_DATA -1 ]:
                    if not aufbereitung [ index_DATA - 2 ]:
                        aufbereitung [ index_DATA - 2 ] = True
                        #print(f" + aufbereitung ")
            
            if not aufbereitung [index_DATA-3]: # rot abziehen
                if not aufbereitung [ index_DATA -1 ]:
                    if aufbereitung [ index_DATA - 2 ]:
                        aufbereitung [ index_DATA - 2 ] = False
                        #print(f" - aufbereitung ")                
             
             
            if aufbereitung [index_DATA- 5]:
                if not counter_flag:
                    counter += 1
                    with open(DATEI, "a", encoding="utf-8") as f:
                        zeitstempel = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"{zeitstempel}  \n")
                counter_flag=True
 
            if not aufbereitung [index_DATA- 5]: # zählen
                counter_flag=False    
             
            index_DATA += 1      
                   
   #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++# detector grapf
                                                                               # werte ausgeben bild löschen vorher hier
            #schwarzes hintergrundbild einmalig                                                                   
            cv2.rectangle( img_DATA , (0, 0) , ( WINDOW_WIDTH_DATA , HEIGHT_DATA ) , (0, 0, 0) , -1) 
            #hier bemalen   
            ihelp=0     
            i=0                                                           
            if 1<2:
                for i in range ( -1 ,  WINDOW_WIDTH_DATA ):
				    # Linie für daten0
                    ihelp = (index_DATA  -i) #index um 800 zurückztellen
                    y0_prev = HEIGHT_DATA        # invertieren, damit 0 unten
                    y0 = data[ihelp, 0] #echte daten auslesen wichtig auch mit anzeige zu tun
                    shelp=y0 
                    datablock=y0
                    schwellwert_vorbereitung = 50
                    cv2.rectangle( img_DATA, (20, 0), (350, 30), (0, 0, 0), -1)            #bild schwarz zurücksetzen hinter test                                               
                    cv2.putText(img_DATA, str(shelp),(30,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0), 2) # textausgabe oben links
                    cv2.putText(img_DATA, str(schwellwert),(130,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255), 2) #textausgabe
                    cv2.putText(img_DATA, str(counter),(230,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255), 2) #textausgabe
                    
                    #print(f" datablock { datablock }  ")
                    
                    cv2.line(img_DATA, (  WINDOW_WIDTH_DATA-i, y0_prev  ), ( WINDOW_WIDTH_DATA-i ,  y0_prev-y0), (0, 255, 0), 1)  # grün grüner ausschlag
                    
                    cv2.line(img_DATA, (WINDOW_WIDTH_DATA -30 , y0_prev - schwellwert ), ( WINDOW_WIDTH_DATA , y0_prev - schwellwert), (0, 0 , 255), 2)  # roter schwellenwertbalkern fest
                    
                    if datablock > schwellwert:  # hier komppt schwellwert                    
                        
                        cv2.line(img_DATA, (WINDOW_WIDTH_DATA-i, schwellwert_vorbereitung), (WINDOW_WIDTH_DATA- i , schwellwert_vorbereitung ), (0, 0, 112), 3)  # rot
                       
                        cv2.line(img_DATA, (WINDOW_WIDTH_DATA-i, 60), (WINDOW_WIDTH_DATA- i , 60 ), (255, 0, 0), 1)  #blau
                        
                        
                        if aufbereitung[ihelp]:
                            cv2.line(img_DATA, (WINDOW_WIDTH_DATA-i, 63), (WINDOW_WIDTH_DATA- i , 63 ), (0, 0, 255), 1)                                      # rot nur bool aus ergebnisarry 
                    else:
                        cv2.line(img_DATA, (WINDOW_WIDTH_DATA-i,  schwellwert_vorbereitung), (WINDOW_WIDTH_DATA- i , schwellwert_vorbereitung ), (112, 0, ), 3)  # blau
                        
    
   																					# anzeigen von allem  
	
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    cartoon_color = cv2.resize(cartoon_color, (End_HD_WIDTH, End_HD_HEIGHT))
    
   
    cv2.namedWindow("Cammax2 Webcam HD", cv2.WINDOW_NORMAL)
    cv2.imshow("Cammax2 Webcam HD", cartoon_color)
   
    cv2.setMouseCallback("Cammax2 Webcam HD", mouse_callback)
    
    if ziel is not None:
        cv2.imshow("Detector", ziel)
    
    if img_DATA is not None:
        cv2.imshow("Detector_Graph", img_DATA)
        cv2.imshow("Detector_Maske", ziel1)
  
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    																				# Frame speichern
    																				
    
        video_frame = color.copy()
        video_frame = cv2.resize(
            output,
            (1980, 1080),
            interpolation=cv2.INTER_AREA
        )

        out.write(video_frame)   #in video datei speichern automatisch letzte datei immer überschrieben 
    
    
    
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                                                                                              #Tastaurbefehle
    
    # Beenden mit 'q'
    
    delay = int(1000 / FPS)
    key = cv2.waitKey(delay) & 0xFF
    
    
    
    # Anzeige wechseln
    
    if key == ord('c'):
        USE_DESKTOP = False
        cap.release()
        cam_index = (cam_index + 1) % len(cameras)
        cap = cv2.VideoCapture(cameras[cam_index], cv2.CAP_V4L2)
        print("🔁 Wechsel zu Kamera:", cameras[cam_index])
	
	# Datei als Eingabe wählen
	 
    if key == ord('f'):
        USE_DESKTOP = False
        USE_VIDEO_FILE =  not USE_VIDEO_FILE
        if USE_VIDEO_FILE:
            VIDEO_PATH = choose_file()
            if not VIDEO_PATH:          # Abbrechen, wenn keine Datei ausgewählt
                USE_VIDEO_FILE = False
            else:
                cap = cv2.VideoCapture(VIDEO_PATH) 
        
        print("Gewählte Datei:")
        print(VIDEO_PATH)           
    
    if key == ord('l'):
        label = label_auswahl()
        
    # Resize von auswal frame intern 
    
    if key == ord('+') or key == ord('='):  # '+' drücken
        if HD_WIDTH < MAX_WIDTH:
           HD_WIDTH = min(HD_WIDTH + 50, MAX_WIDTH)
           HD_HEIGHT = int(HD_WIDTH / ASPECT_RATIO)
           print(f"Auflösung erhöht: {HD_WIDTH}x{HD_HEIGHT}")

    if key == ord('-') or key == ord('_'):  # '-' drücken
         if HD_WIDTH > MIN_WIDTH:
           HD_WIDTH = max(HD_WIDTH - 50, MIN_WIDTH)
           HD_HEIGHT = int(HD_WIDTH / ASPECT_RATIO)
           print(f"Auflösung reduziert: {HD_WIDTH}x{HD_HEIGHT}")
           # bug rezis minus
           x=100
           y=100
           x =   (x * ASPECT_RATIO)
           y =   (y / ASPECT_RATIO)
           x=int(x)
           y=int(y)
           x1, y1 = x-25, y-25
           x2, y2 = x+25, y+25

    
    # Pfeiltasten abfangen
    
    # Helligkeit
    
    if key == 27:  # ESC
        print("Programm beendet")
        break
            
    elif key == 82:  # UP
        
        v_offset += brightness_step
        if v_offset>255-brightness_step:
            v_offset=255
        print(f": {v_offset} +Helligikeit ")
    elif key == 84:  # DOWN
        if v_offset < brightness_step:
            v_offset =  brightness_step
        v_offset -= brightness_step
        print(f": {v_offset} -Helligikeit")
   
   
   # Sättigung
   
    elif key == 81:  # LEFT
        if s_scale<0.1:
            s_scale=0.1
        s_scale = max(0.0, s_scale - 0.1)
        print(f": {s_scale} -Sättigung")
    elif key == 83:  # RIGHT
        if s_scale>259:
              s_scale=259
        s_scale = min(3.0, s_scale + 0.1)
        print(f": {s_scale} +Sättigung")
   
   # Kippschalter für Modi
   
    elif key == ord('1'):
        kippschalterf1+=1
        print(f"{ kippschalterf1 } Anzeige ")
    
    
    
    # Für Threshold
    elif key == ord('6'):
        my_thresh_thresh-=1
        print(f"{ my_thresh_thresh } THRESH -(Schwellwert) ")
    elif key == ord('7'):
        my_thresh_thresh+=1
        print(f"{ my_thresh_thresh } THRESH +(Schwellwert) ")
    if my_thresh_thresh < 0  :
        my_thresh_thresh = 0
    if my_thresh_thresh > 255  :
        my_thresh_thresh=255
   
   
    # Für schwellenwert
    elif key == ord('8'):
        schwellwert-=1
        print(f"{ schwellwert }  -(Schwellwert) ")
    elif key == ord('9'):
        schwellwert+=1
        print(f"{ schwellwert }  +(Schwellwert) ")
    
    
    elif key == ord('0'):
        counter=0
        with open(DATEI, "w", encoding="utf-8"):
            pass  # Datei wird geöffnet und überschrieben sofort wieder geschlossen, bleibt leer
        print(f"{ counter }  +(counter) ")
    
    elif key == ord('p'):
        drawing_pin  = not drawing_pin
        drawing  = drawing_pin
        print(f"Set Detector   { drawing_pin }   ")

    elif key == ord('s'):
        speichern_pin  = not speichern_pin
        print(f"Set speicher   { speichern_pin }   ")
        
    elif key == ord('v'):
        
        VIDEO_PATH  = choose_file2()
        print(f"Viedeo Ordner geöffnet   ")
        USE_DESKTOP = False
        
        if VIDEO_PATH:
        
            endung = os.path.splitext(VIDEO_PATH)[1].lower()
        
            if endung in [".jpg", ".jpeg", ".png", ".bmp"]:
        
                image_frame = cv2.imread(VIDEO_PATH)
        
                if image_frame is not None:
                    USE_IMAGE_FILE = True
                    USE_VIDEO_FILE = False
                    print("Bild als Stream geladen:", VIDEO_PATH)
        
            else:
        
                USE_IMAGE_FILE = False
                cap = cv2.VideoCapture(VIDEO_PATH)
                print("Video geladen:", VIDEO_PATH)  

        #if speichern_pin == False:
            #label = label_auswahl()
    
    elif key == ord('d'):
    
        MONITOR_INDEX += 1
    
        if MONITOR_INDEX > 2:
            MONITOR_INDEX = 0
    
        USE_DESKTOP = True

        #print(
        #    "Monitor gewechselt:",
        #    MONITOR_INDEX
        #)
        
                 
        if schwellwert < 0  :
            schwellwert = 0
        if schwellwert > 2500  :
            schwellwert=2500
   
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   
    # Aufräumen

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video gespeichert in: {os.path.abspath(OUTPUT_PATH)}")
print("---")
csv_status()
print(f"---------------------------------------------------------------------------------------------------End-")


#print(f" datablock { datablock }  ")
#print(f" datablock { datablock }  ")
#print(f" if datablock { datablock }  . y0 { y0 }  . ihelp = { ihelp } . i = { i }. index_DATA = { index_DATA } .data[i, 0] = { data[index_DATA+i, 0] } . anzahl_W = { anzahl_W }  ")
                   

