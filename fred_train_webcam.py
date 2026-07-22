# fred_train_webcam.py

import cv2
import torch
import torch.nn as nn

import config

from neuralNetwork import NeuralNetwork


# ============================
# Gerät
# ============================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Gerät:", device)


# ============================
# Netzwerk aus config
# ============================

net = NeuralNetwork(
    config.LAYERS,
    config.ACTIVATIONS,
    config.DROPOUT
)


net.to(device)

net.train()


print(net)


# ============================
# Training
# ============================

criterion = nn.MSELoss()


optimizer = torch.optim.Adam(
    net.parameters(),
    lr=config.LEARNING_RATE
)


# ============================
# Webcam
# ============================

cap = cv2.VideoCapture(0)


print(
    "Webcam:",
    cap.isOpened()
)


print()
print("Steuerung:")
print("SPACE = aktuelles Bild lernen")
print("q     = Ende")


# ============================
# Training Schleife
# ============================

bilder_gelernt = 0


while True:


    ret, frame = cap.read()


    if not ret:
        break



    cv2.imshow(
        "Fred Webcam Training ohne Label",
        frame
    )


    taste = cv2.waitKey(1) & 0xff



    if taste == ord("q"):

        break



    if taste == 32:


        print(
            "Trainiere Bild:",
            bilder_gelernt
        )


        # ============================
        # Bild vorbereiten
        # ============================

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
            1,
            config.INPUT_SIZE
        )


        tensor = tensor.to(device)



        # ============================
        # Lernen ohne Label
        # ============================

        optimizer.zero_grad()



        ausgabe = net(
            tensor
        )



        # Eingabe = Ziel
        loss = criterion(
            ausgabe,
            tensor
        )



        loss.backward()


        optimizer.step()



        print(
            "Loss:",
            loss.item()
        )


        bilder_gelernt += 1



        # speichern

        torch.save(
            net.state_dict(),
            "fred_webcam_auto.pth"
        )



cap.release()

cv2.destroyAllWindows()


print()
print(
    "Fertig. Bilder gelernt:",
    bilder_gelernt
)