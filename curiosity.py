import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from threading import Thread
from time import sleep

class Autoencoder(nn.Module):
    def __init__(self, procesimgsize=64, params=32, size=6, size2=4):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, params, kernel_size=size, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=size2, stride=size2, padding=0),
            nn.Conv2d(params, params, kernel_size=size, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=size2, stride=size2, padding=0)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(params, params, kernel_size=size, stride=1, padding="same"),
            nn.ReLU(),
            nn.Upsample(scale_factor=size2, mode="nearest"),
            nn.Conv2d(params, params, kernel_size=size, stride=1, padding="same"),
            nn.ReLU(),
            nn.Upsample(scale_factor=size2, mode="nearest"),
            nn.Conv2d(params, 1, kernel_size=size, stride=1, padding="same"),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class curiosity:
    procesimgsize = 64
    saved_model_uri = "saved_model.pth"
    split_values = [[0, 0], [0, 0]]
    pause=0

    def __init__(self, camera, pause=0,savemodel=True):
        self.pause=pause
        self.state_vals=[]
        for s in self.split_values:
            if type(s)==list:
                for ss in s:
                    self.state_vals.append(0)
            else:
                self.state_vals.append(0)
        print("self.state_vals",self.state_vals)
        self.savemodel = savemodel
        self.CAM = camera
        self.ready = False
        self.autoencoder = Autoencoder(self.procesimgsize)
        self.autoencoder.train()
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

        if os.path.isfile(self.saved_model_uri):
            self.autoencoder.load_state_dict(torch.load(self.saved_model_uri))
            self.autoencoder.eval()
        else:
            print("No saved model found. Starting with a new model.")

        self.ready = True
        sleep(1)
        self.new_image = False

    def preprocess_image(self, new_image):
        """
        Splits the input image into blocks based on `self.split_values`.
        """
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        height, width = new_image.shape

        # Normalize split_values to determine rows and columns
        # For example: [[0, 0]] -> [[2, 2]]
        split_values = [
            [x if x != 0 else 2, y if y != 0 else 2]  # Default to a 2x2 grid if 0
            for x, y in self.split_values
        ]

        image_blocks = []

        # Assuming split_values always contains one pair for now
        row_split, col_split = split_values[0]  # Take the first pair

        # Calculate block dimensions
        block_height = height // row_split
        block_width = width // col_split

        for row in range(row_split):
            for col in range(col_split):
                # Extract each block
                block = new_image[
                    row * block_height : (row + 1) * block_height,
                    col * block_width : (col + 1) * block_width,
                ]

                # Resize to processing size and normalize
                block = cv2.resize(block, (self.procesimgsize, self.procesimgsize))
                block = torch.tensor(block, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

                # Append to list
                image_blocks.append(block)

        return image_blocks



    def predict_and_calculate_mse(self, image):
        with torch.no_grad():
            decoded_image = self.autoencoder(image)
            mse = torch.mean((image - decoded_image) ** 2).item()
        
        return mse

    def update_model_with_new_image(self, image, epochs=5):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.autoencoder(image)
            loss = self.criterion(output, image)
            loss.backward()
            self.optimizer.step()

    def run_curiosity(self):
        self.new_image = self.CAM.frame

        # Preprocess the image into blocks
        blocks = self.preprocess_image(self.new_image)
        

        # Initialize list for MSE values
        mse_values = []

        # Predict and calculate MSE for each block
        for block in blocks:
            mse = self.predict_and_calculate_mse(block)
            mse_values.append(mse * 1000)  # Scale MSE for readability

        # Generate summary of MSE values
        mse_summary = " ".join([f"B{i+1}:{mse:.2f}" for i, mse in enumerate(mse_values)])
        #mse_sum = sum(mse_values)
        #mse_summary += f" SUM:{mse_sum:.2f}"

        self.state_vals = mse_values  # Store MSE values for the blocks
        self.CAM.curiosity_data = self.state_vals
        #print(len(mse_values))
        # Update the model with all blocks at the end
        for block in blocks:
            self.update_model_with_new_image(block, epochs=5)  # Use 1 epoch per block to avoid overfitting
        sleep(self.pause)
    def curiosity_process(self):
        while True:
            if self.ready:
                self.run_curiosity()
            else:
                sleep(1)

    def start(self):
        tc = Thread(target=self.curiosity_process)
        tc.start()

    def end(self):
        if self.savemodel:
            torch.save(self.autoencoder.state_dict(), self.saved_model_uri)
