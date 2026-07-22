#visualizer
import matplotlib.pyplot as plt
import torch

import config

import cv2


class TrainingVisualizer:

    def __init__(self):

        plt.ion()

        self.loss_history = []
        self.acc_history = []

        # Ein großes Fenster
        self.fig, self.ax = plt.subplots(
            2,
            2,
            figsize=(12, 8)
        )

        self.fig.canvas.manager.set_window_title(
            "Fred PyTorch Training"
        )

        self.fig.tight_layout(
            pad=3
        )


    def update(
        self,
        loss,
        accuracy,
        progress
    ):

        self.loss_history.append(
            loss
        )

        self.acc_history.append(
            accuracy
        )


        # Loss
        self.ax[0,0].clear()

        self.ax[0,0].plot(
            self.loss_history
        )

        self.ax[0,0].set_title(
            "Loss"
        )


        # Accuracy
        self.ax[0,1].clear()

        self.ax[0,1].plot(
            self.acc_history
        )

        self.ax[0,1].set_title(
            "Accuracy"
        )


        # Status
        self.ax[1,1].clear()

        self.ax[1,1].text(
            0.1,
            0.7,
            f"Fortschritt: {progress:.1f} %",
            fontsize=14
        )

        self.ax[1,1].text(
            0.1,
            0.5,
            f"Loss: {loss:.5f}",
            fontsize=14
        )

        self.ax[1,1].text(
            0.1,
            0.3,
            f"Accuracy: {accuracy:.2f} %",
            fontsize=14
        )

        self.ax[1,1].axis(
            "off"
        )


        self.fig.canvas.draw_idle()

        self.fig.canvas.flush_events()



    def show_training_image(
        self,
        image,
        label,
        prediction,
        output
    ):
        #print("SHOW_TRAINING_IMAGE AUFGERUFEN")
        image = image.detach().cpu()


        # Bild anzeigen
        self.ax[1,0].clear()

        bild = image.reshape(
            config.input_y,
            config.input_x
        )
        #print("RAND OBEN:", bild[0,:20])
        #print("RAND UNTEN:", bild[-1,:20])
        #print("BILD FORM:", bild.shape)
        #print("BILD ERSTE ZEILE:")
        #print(bild[0,0:50])
        
        #bild = bild.T
        
        self.ax[1,0].imshow(
            bild,
            cmap="gray",
            aspect="equal",
            vmin=0,
            vmax=1
        )      

        self.ax[1,0].axis(
            "off"
        )

        for spine in self.ax[1,0].spines.values():
            spine.set_visible(False)
            
        # Wahrscheinlichkeit berechnen
        probability = torch.softmax(
            output.detach(),
            dim=0
        )


        confidence = (
            probability[prediction]
            .item()
            *
            100
        )


        if label == prediction:
            status = "RICHTIG"
        else:
            status = "FALSCH"


        self.ax[1,0].set_title(
            f"Label: {label}\n"
            f"Vorhersage: {prediction}\n"
            f"Sicherheit: {confidence:.2f}%\n"
            f"{status}"
        )


        self.fig.canvas.draw_idle()

        self.fig.canvas.flush_events()
        
        
        
        
        
        
        
        
        