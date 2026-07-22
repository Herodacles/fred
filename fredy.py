# fredy.py
import sys
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk

import torch

from neuralNetwork import NeuralNetwork
import config
KLASSEN = config.KLASSEN
# ============================
# Video
# ============================


#if len(sys.argv) > 1:
#    VIDEO_DATEI = sys.argv[1]
#else:
#    VIDEO_DATEI = "test.mp4"


#cap = cv2.VideoCapture(
#    VIDEO_DATEI
#)

#print("Video:", VIDEO_DATEI)
#print("Video geöffnet:", cap.isOpened())




# ============================
# Webcam öffnen
# ============================

KAMERA = 0      # erste Webcam
# KAMERA = 1    # zweite Webcam

cap = cv2.VideoCapture(KAMERA)

print("Webcam geöffnet:", cap.isOpened())


# ============================
# Netzwerk laden
# ============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


net = NeuralNetwork(
    config.LAYERS,
    config.ACTIVATIONS,
    config.DROPOUT
)


net.load_state_dict(
    torch.load(
        "fred_model.pth",
        map_location=device
    )
)


net.to(device)
net.eval()


print("Fred geladen")


# ============================
# Erkennung
# ============================

def video_erkennen():

    ret, frame = cap.read()

    if not ret:
        print("Kein Kamerabild")
        fenster.after(30, video_erkennen)
        return
    
    '''    
    if not ret:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            0
        )
        fenster.after(
            30,
            video_erkennen
        )
        return
    '''

    # Eingabe für Netz

    bild = cv2.resize(
        frame,
        (
            config.input_x,
            config.input_y
        )
    )


    grau = cv2.cvtColor(
        bild,
        cv2.COLOR_BGR2GRAY
    )


    daten = grau / 255.0


    tensor = torch.tensor(
        daten,
        dtype=torch.float32
    )


    tensor = tensor.reshape(
        config.input_x *
        config.input_y
    )


    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)


    with torch.no_grad():

        output = net(
            tensor
        )


        probs = torch.softmax(
            output,
            dim=1
        )


        wert = torch.argmax(
            probs,
            dim=1
        )


    klasse = wert.item()

    ergebnis.config(
        text=
        f"Fred erkennt: {KLASSEN[klasse]}\n"
        f"Klasse: {klasse}\n"
        f"Sicherheit: {probs[0][wert].item()*100:.1f}%"
    )

    # Video in Tk anzeigen

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    img = Image.fromarray(
        rgb
    )


    img = img.resize(
        (
            640,
            360
        )
    )


    photo = ImageTk.PhotoImage(
        img
    )


    video_label.config(
        image=photo
    )

    video_label.image = photo


    fenster.after(
        30,
        video_erkennen
    )


# ============================
# Fenster
# ============================


fenster = tk.Tk()

fenster.title(
    "Fredy Video Erkennung"
)


video_label = tk.Label(
    fenster
)

video_label.pack()


ergebnis = tk.Label(
    fenster,
    text="Starte Video...",
    font=("Arial",18)
)

ergebnis.pack()


video_erkennen()


fenster.mainloop()


cap.release()









































