#neural network.py
import torch
import torch.nn as nn

import importlib

import config
from config import *

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

############################################################################################



    
 
class TrainingHistory:

    def __init__(self):
        self.clear()


    def clear(self):

        self.epoch = []
        self.batch = []

        self.loss = []
        self.accuracy = []

        self.learning_rate = []

        self.prediction = []
        self.label = []

        # neue Analysewerte
        self.confidence = []
        self.output_probability = []

        self.wrong_images = []
        self.wrong_labels = []
        self.wrong_predictions = []
        self.wrong_probability = []

        self.batch_time = []
        self.gpu_memory = []

   

        self.time = []


    def add(
        self,
        epoch,
        batch,
        loss,
        accuracy,
        learning_rate=0.0,
        prediction=None,
        label=None,
        confidence=None,
        output_probability=None,
        hidden_activation=None
    ):

        self.epoch.append(epoch)
        self.batch.append(batch)

        self.loss.append(float(loss))
        self.accuracy.append(float(accuracy))

        self.learning_rate.append(
            float(learning_rate)
        )

        self.prediction.append(prediction)
        self.label.append(label)


        # neue Daten speichern

        self.confidence.append(
            confidence
        )

        self.output_probability.append(
            output_probability
        )

      


    def statistics(self):

        print()
        print("========== Training ==========")

        print(
            "Schritte:",
            len(self.loss)
        )


        if len(self.loss):

            print(
                "Min Loss:",
                np.min(self.loss)
            )

            print(
                "Max Loss:",
                np.max(self.loss)
            )

            print(
                "Ø Loss:",
                np.mean(self.loss)
            )


            print()

            print(
                "Min Accuracy:",
                np.min(self.accuracy)
            )

            print(
                "Max Accuracy:",
                np.max(self.accuracy)
            )

            print(
                "Ø Accuracy:",
                np.mean(self.accuracy)
            )


        print("==============================")


    def live_plot(
        self,
        bild=None,
        label=None,
        prediction=None,
        probabilities=None,
        activations=None,
        net=None
    ):
    
        #print("LIVE_PLOT AUS DATEI:", __file__)


        plt.ion()


        if not hasattr(self, "live_fig"):


            self.live_fig, self.live_ax = plt.subplots(
                3,
                6,
                figsize=(24,12)
            )
            self.live_fig.canvas.manager.set_window_title(
                "Fred Live Neural Network Manager"
            )
            
            plt.show(block=False)



        for row in self.live_ax:

            for ax in row:

                ax.clear()



        # ----------------------
        # Zeile 1
        # ----------------------


        self.live_ax[0,0].plot(
            self.loss
        )

        self.live_ax[0,0].set_title(
            "Loss"
        )


        self.live_ax[0,1].scatter(
            range(len(self.loss)),
            self.loss,
            s=5
        )

        self.live_ax[0,1].set_title(
            "Loss Punkte"
        )


        self.live_ax[0,2].plot(
            self.accuracy
        )

        self.live_ax[0,2].set_title(
            "Accuracy"
        )


        self.live_ax[0,3].scatter(
            range(len(self.accuracy)),
            self.accuracy,
            s=5
        )

        self.live_ax[0,3].set_title(
            "Accuracy Punkte"
        )


        self.live_ax[0,4].plot(
            self.learning_rate
        )

        self.live_ax[0,4].set_title(
            "Learning Rate"
        )


        if len(self.accuracy):

            fehler = [
                100-x*100
                for x in self.accuracy
            ]

            self.live_ax[0,5].plot(
                fehler
            )

        self.live_ax[0,5].set_title(
            "Fehler %"
        )
        self.live_ax[2,0]
        self.live_ax[2,1]
        self.live_ax[2,2]
        self.live_ax[2,3]
        self.live_ax[2,4]
        self.live_ax[2,5]

        # ----------------------
        # Zeile 2
        # ----------------------


        if bild is not None:

            self.live_ax[1,0].imshow(
                bild.reshape(config.input_y,config.input_x),
                cmap="gray"
            )

            self.live_ax[1,0].set_title(
                f"Echt {label} | KI {prediction}"
            )

        if probabilities is not None:

            werte = probabilities

            #print("OUTPUT_SIZE:", config.OUTPUT_SIZE)
            #print("len(werte):", len(werte))
            
            self.live_ax[1,1].bar(
                range(config.OUTPUT_SIZE),
                werte
            )

            for nummer, wert in enumerate(werte):

                self.live_ax[1,1].text(
                    nummer,
                    wert,
                    f"{wert:.2f}",
                    ha="center",
                    fontsize=8
                )

            self.live_ax[1,1].set_title(
                "Wahrscheinlichkeit 0-9"
            )

            self.live_ax[1,1].set_xlabel(
                "Zahl"
            )

            self.live_ax[1,1].set_ylabel(
                "Wert"
            )

            self.live_ax[1,1].set_ylim(
                0,
                1
            )    
    
            self.live_ax[1,2].set_title(
                "Hidden Layer 1"
            )
    
    
            self.live_ax[1,3].set_title(
                "Hidden Layer 2"
            )
    
    
            self.live_ax[1,4].set_title(
                "Gewichte"
            )
    
    
            self.live_ax[1,5].set_title(
                "Gradienten"
            )


        # ----------------------
        # Zeile 3
        # ----------------------


        self.draw_network(
            self.live_ax[2,0],
            net,
            activations
        )  
        
        self.live_ax[2,1].set_title(
            "Fehlerbilder"
        )

        self.live_ax[2,2].set_title(
            "Batch Zeit"
        )

        self.live_ax[2,3].set_title(
            "GPU Speicher"
        )

        self.live_ax[2,4].set_title(
            "Speicher"
        )

        self.live_ax[2,5].set_title(
            "Zeit"
        )

        
        self.live_ax[1,5].text(
            0.1,
            0.7,
            f"Label: {label}",
            fontsize=14
        )
        
        self.live_ax[1,5].text(
            0.1,
            0.5,
            f"Vorhersage: {prediction}",
            fontsize=14
        )
        
        if label == prediction:
            status = "RICHTIG"
        else:
            status = "FALSCH"
        
        self.live_ax[1,5].text(
            0.1,
            0.3,
            status,
            fontsize=16
        )
        
        self.live_ax[1,5].axis(
            "off"
        )
        



        self.live_fig.tight_layout()

        if probabilities is not None:
        
            self.live_ax[2,5].clear()
        
            self.live_ax[2,5].bar(
                range(config.OUTPUT_SIZE),
                probabilities
            )
        
            self.live_ax[2,5].set_title(
                "Output 0-9"
            )

        self.live_fig.canvas.draw()

        self.live_fig.canvas.flush_events()
        
        
        
    def draw_network(
        self,
        ax,
        net,
        activations=None
    ):
    
        schichten = []
    
        for layer in net.layers:
    
            if isinstance(layer, nn.Linear):
        
                if len(schichten) == 0:
        
                    schichten.append(
                        layer.in_features
                    )
        
                schichten.append(
                    layer.out_features
                )
            
        
        
    
        ax.clear()
    
    
    
    
        for x, anzahl in enumerate(schichten):
    
            anzeigen = anzahl
            
            
            if activations is not None:
            
                if x < len(activations):
            
                    anzeigen = min(
                        anzeigen,
                        activations[x].shape[1]
                    )
    
            y = np.linspace(
                -1,
                1,
                anzeigen
            )
    
    
            groesse = np.ones(
                anzeigen
            ) * 20
    
    
            if activations is not None:
    
                if x < len(activations):
    
                    werte = activations[x][0]
    
                    werte = werte.numpy()


                    werte = werte[:anzeigen]

                    
                    if x != 0:
                    
                        werte = np.abs(werte)
                    
                    
                    if x == 0:
                    
                        # Eingang Pixel 0-255 auf 0-1 bringen
                        if werte.max() > 1:
                    
                            werte = werte / 255.0
                    
                    
                    else:
                    
                        # Hidden Aktivierungen normieren
                        if werte.max() > 0:
                    
                            werte = werte / werte.max()
                    
                    
                    
                    groesse = np.zeros(
                        anzeigen
                    )
                   
                    # kontinuierliche Skalierung

                    min_groesse = 1
                    max_groesse = 100
                    
                    
                    for i, wert in enumerate(werte):
                    
                        groesse[i] = (
                            min_groesse
                            +
                            wert * (max_groesse - min_groesse)
                        )
                        
            ax.scatter(
                [x] * anzeigen,
                y,
                s=groesse
            )
    
    
            ax.text(
                x,
                -1.2,
                str(anzahl),
                ha="center"
            )
    
    
            
    
        ax.set_title(
            "Neuronales Netzwerk"
        )
    
    
        ax.axis("off")
        
############################################################################################


class NeuralNetwork(nn.Module):

    def __init__( self, layers, activations=None, dropout=0.0 ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.history = TrainingHistory()

        # Schichten erzeugen
        for i in range(len(layers)-1):

            # Linear Layer
            self.layers.append(
                nn.Linear(
                    layers[i],
                    layers[i+1]
                )
            )


            # keine Aktivierung nach letzter Schicht
            if i < len(layers)-2:

                if activations:

                    self.layers.append(
                        self.get_activation(
                            activations[i]
                        )
                    )


                if dropout > 0:

                    self.layers.append(
                        nn.Dropout(dropout)
                    )


    def get_activation(self, name):

        if name == "ReLU":
            return nn.ReLU()

        elif name == "LeakyReLU":
            return nn.LeakyReLU()

        elif name == "PReLU":
            return nn.PReLU()

        elif name == "ELU":
            return nn.ELU()

        elif name == "SELU":
            return nn.SELU()

        elif name == "GELU":
            return nn.GELU()

        elif name == "SiLU":
            return nn.SiLU()

        elif name == "Tanh":
            return nn.Tanh()

        elif name == "Sigmoid":
            return nn.Sigmoid()

        elif name == "Softplus":
            return nn.Softplus()

        else:
            raise ValueError(
                "Unbekannte Aktivierung: "
                + name
            )


    def backquery(
        self,
        zielklasse,
        staerke=1.0
    ):

        device = next(self.parameters()).device


        # Output-Wunsch erzeugen
        x = torch.zeros(
            1,
            self.layers[-1].out_features,
            device=device
        )


        x[0, zielklasse] = staerke


        # Rückwärts durch Linear Layer
        for layer in reversed(self.layers):

            if isinstance(layer, nn.Linear):

                gewicht = layer.weight

                x = torch.matmul(
                    x,
                    gewicht
                )


                # normieren
                x = (
                    x - x.min()
                ) / (
                    x.max() - x.min() + 1e-8
                )


        return x.detach().cpu()




    def forward(self, x):
    
        self.activations = []
    
        for layer in self.layers:
    
            x = layer(x)
    
    
            if isinstance(
                layer,
                nn.Linear
            ):
    
                self.activations.append(
                    x.detach().cpu()
                )
    
    
        return x  
    
    
    
#######################################################################################    
    
    
    
    
    