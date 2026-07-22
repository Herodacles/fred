# fred.py
import os
import sys
import numpy as np
import scipy.special

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from visualize import TrainingVisualizer
import pandas

import torch
import torch.nn as nn

from neuralNetwork import NeuralNetwork
import neuralNetwork

print("Geladene Datei:")
print(neuralNetwork.__file__)

import config

from dataset import FredDataset
from torch.utils.data import DataLoader



traindateiname = "mnist_dataset/fredy_train.csv"
if len(sys.argv) > 1:
    h1="mnist_dataset/"
    h2=sys.argv[1]
    h3=".csv"
    traindateiname = h1+h2+h3

print("CSV Trainigsdatei:", traindateiname)


#=====================================
# GPU auswählen
# =====================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Gerät:", device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.memory_allocated()/1024**2, "MB")
else:
    print("Keine GPU aktiv")
print("---")





#################################################

# =====================================
# Eigene CSV laden hier training mit label datei
# =====================================

dataset = FredDataset(
    traindateiname
)

loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=True,
    persistent_workers=True
)

 #=====================================
# Testdaten laden
# =====================================

test_dataset = FredDataset(
    traindateiname
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)

#bilder, labels = next(iter(loader))
bilder = next(iter(loader))

print("Bilder Anzahl:", len(bilder))

#print("Bilder:", bilder.shape)
#print("Labels:", labels.shape)
print("---")

import pandas as pd

df = pd.read_csv(
    traindateiname,
    header=None
)


print("CSV TEST")
print(df.shape)
print(df.iloc[0,0:20])

print(df.shape)
print(df.iloc[0])
print("---")

# =====================================
# Netzwerk erzeugen
# =====================================

net = NeuralNetwork(
    config.LAYERS,
    config.ACTIVATIONS,
    config.DROPOUT
)

print("History vorhanden:", hasattr(net, "history"))
print(net.__dict__.keys())

net = net.to(device)

print(net)


MODEL_FILE = "fred_model.pth"

if os.path.exists(MODEL_FILE):

    print("Modell gefunden - lade Gewichte")

    net.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location=device
        )
    )

    net.to(device)

    print("Modell geladen")

else:

    print("Kein Modell vorhanden - Training startet")

    # hier kommt dein Trainingscode hin

    torch.save(
        net.state_dict(),
        MODEL_FILE
    )

    print("Modell gespeichert")


#####


# =====================================
# Backquery alle Outputs anzeigen
# =====================================

def zeige_backquery_alle_outputs(net):

    fig, ax = plt.subplots(4,5, figsize=(12,8))

    fig.canvas.manager.set_window_title(
        "Fred Backquery - alle Outputs"
    )

    for zahl in range(config.OUTPUT_SIZE):

        bild = net.backquery(
            zahl
        )

        bild = bild.reshape(
            config.input_x,
            config.input_y
        )

        achse = ax[zahl //5, zahl % 5]

        achse.imshow(
            bild,
            cmap="gray"
        )

        achse.set_title(
            f"Output {zahl}"
        )

        achse.axis(
            "off"
        )


    plt.tight_layout()

    plt.show()









# =====================================
# Training vorbereiten
# =====================================

#visualizer = TrainingVisualizer()

criterion = nn.CrossEntropyLoss()

#criterion = nn.MSELoss()


optimizer = torch.optim.Adam(
    net.parameters(),
    lr=0.003
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

epochs = config.EPOCHS

#visualizer = TrainingVisualizer()

# =====================================
# Training
# =====================================

for epoch in range(epochs):
    
    
    net.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_loss = 0
    richtig = 0
    gesamt = 0

    for batch, (bilder, labels) in enumerate(loader):
        
        bilder = bilder.to(device)
        labels = labels.to(device)
        #print("Bilder Shape:", bilder.shape)
        output = net(bilder)
        
        #print("Merkmale:", output[0].detach().cpu().numpy())
        
        #loss = criterion(
        #    output,
        #    torch.mean(bilder, dim=1, keepdim=True)
        #)
        
        loss = criterion(
            output,
            labels
        )
        
                
        optimizer.zero_grad()
        loss.backward()
        
               
        vorhersage = torch.argmax(output, dim=1)
       
        
        batch_accuracy = 0
        
        optimizer.step()
        
        
        
        net.history.add(
            epoch,
            batch,
            loss.item(),
            batch_accuracy,
            optimizer.param_groups[0]["lr"]
        )
        
        
        
                
        wahrscheinlichkeit = torch.softmax(
            output,
            dim=1
            ).detach().cpu().numpy()[0]
    
        accuracy = 0
        
        #richtig += (vorhersage == labels).sum().item()

        #gesamt += labels.size(0)

        total_loss += loss.item()
        
           
        

        for i in range(len(bilder)):


            
                     




            wahrscheinlichkeiten_i = torch.softmax(
                output[i],
                dim=0
            ).detach().cpu().numpy()
        
            
            
            aktivierungen = [
               bilder[i].cpu().unsqueeze(0)
            ]
            
            for activation in net.activations:
            
               aktivierungen.append(
                   activation[i].unsqueeze(0)
               )                  
             
                               
            if False:
            
                net.history.wrong_images.append(
                    bilder[i].cpu()
                )
            
                #net.history.wrong_labels.append(
                #    labels[i].item()
                #)
            
                net.history.wrong_predictions.append(
                    vorhersage[i].item()
                )
            
                net.history.wrong_probability.append(
                    wahrscheinlichkeiten_i
                )
                        
                
            
              
                        
        net.train()        
        
        
        print("LIVE_PLOT START")
        
        
        if batch % 60 == 0:
                        
                net.eval()
                         
                        
                        
                net.history.live_plot(
                    bilder[i].cpu(),
                    labels[i].item(),
                    vorhersage[i].item(),
                    wahrscheinlichkeiten_i,
                    aktivierungen,
                    net
                )         
                
                #visualizer.show_training_image(
                #    bilder[i].cpu(),
                #    -1,
                #    vorhersage[i].item(),
                #    output[i]
                #)
                        
        
    if gesamt > 0:
        genauigkeit = richtig / gesamt * 100
    else:
        genauigkeit = 0
    print(
        "Epoch:",
        epoch + 1,
        "Loss:",
        total_loss / len(loader),
        "Genauigkeit:",
        genauigkeit,
        "%"
    )
    print(
        "Gesammelte Bilder:",
        len(net.history.label)
    )   

    scheduler.step(
        total_loss / len(loader)
    )
    
    print(
    "Learning Rate:",
    optimizer.param_groups[0]["lr"]
    )

    torch.save(
        net.state_dict(),
        MODEL_FILE
        )


zeige_backquery_alle_outputs(net)


print("Modell gespeichert")

print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

# =====================================
# Loss Punktewolke speichern
# =====================================

plt.figure(
    figsize=(10,5)
)

plt.scatter(
    range(len(net.history.loss)),
    net.history.loss,
    s=3
)

plt.title(
    "Loss Punktewolke"
)

plt.xlabel(
    "Trainingsschritt"
)

plt.ylabel(
    "Loss"
)

plt.grid(
    True
)

plt.savefig(
    "loss_punktewolke.png",
    dpi=300
)

plt.show()

print(
    "Loss Punktewolke gespeichert: loss_punktewolke.png"
)


net.history.clear()

sys.exit()



