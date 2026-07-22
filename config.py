#config.py
# =====================================
# Fred PyTorch Netzwerk Konfiguration
# =====================================


input_x= 240
input_y=130



# Eingabe / Ausgabe
INPUT_SIZE = input_x * input_y






KLASSEN = {
    0:"Menschen",
    1:"Tiere",
    2:"Pflanzen",
    3:"Pilze",
    4:"Bakterien",
    5:"Einzeller",
    6:"Insekten",
    7:"Fische",
    8:"Vögel",
    9:"Säugetiere",
    10:"Reptilien",
    11:"Amphibien",
    12:"Gesteine",
    13:"Mineralien",
    14:"Metalle",
    15:"Wasser",
    16:"Wolken/Wetter",
    17:"Landschaften",
    18:"Gebäude",
    19:"Fahrzeuge"
}

ACTIVATIONS = [
    "Sigmoid",
    "Sigmoid"
]


OUTPUT_SIZE = len(KLASSEN)

# Netzaufbau

LAYERS = [
    INPUT_SIZE,
   177,
   40,
   OUTPUT_SIZE
]

# =====================================
# Aktivierungsfunktionen PyTorch
# =====================================

# ReLU       -> Standard, schnell, häufig verwendet
# LeakyReLU  -> ReLU mit kleinem negativen Anteil
# PReLU      -> LeakyReLU mit lernbarem Faktor
# ELU        -> Glatter Übergang, gute Alternative zu ReLU
# SELU       -> Selbstnormalisierende Netzwerke
# GELU       -> Modern, häufig bei Transformern
# SiLU       -> Auch Swish genannt, moderne Aktivierung
# Tanh       -> Wertebereich -1 bis +1
# Sigmoid    -> Wertebereich 0 bis 1
# Softplus   -> Glatte ReLU-Variante
# Softsign   -> Ähnlich wie Tanh
# Hardtanh   -> Begrenzte Tanh-Variante
# Softmax    -> Wahrscheinlichkeiten für mehrere Klassen
# Aktivierungen pro Hidden-Schicht


BATCH_SIZE =2048
num_workers=6 # muss 1 sein

# Gerät
USE_GPU = True


EPOCHS = 50


# Training
LEARNING_RATE = 0.001

# Dropout
DROPOUT = 0.2









































