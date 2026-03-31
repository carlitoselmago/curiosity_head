import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from threading import Thread
from time import sleep

class Autoencoder(nn.Module):
    def __init__(self, procesimgsize=32, params=12, size=6, size2=4):    
    #def __init__(self, procesimgsize=64, params=32, size=6, size2=4):
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
    
    pause=0

    def __init__(self, camera, pause=0,split_values = [1,1],savemodel=True, visualization_mode="heatmap", movement_grid=None):
        self.split_values=split_values
        self.pause=pause
        self.visualization_mode = visualization_mode
        self.movement_grid = movement_grid if movement_grid is not None else split_values
        self.visual_smoothing_alpha = 0.25
        self.previous_visualization_map = None
        movement_cols, movement_rows = self.movement_grid
        self.state_vals = [0] * (movement_cols * movement_rows)
       
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
        Splits the input image into blocks and keeps the metadata required to
        rebuild a full-size error map after block inference.
        """
        grayscale_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        height, width = grayscale_image.shape

        # Validate split_values
        if not isinstance(self.split_values, list) or len(self.split_values) != 2:
            raise ValueError("split_values must be a list of two integers: [columns, rows]")

        cols, rows = self.split_values

        # Default to single block if values are invalid
        if cols <= 0:
            cols = 1
        if rows <= 0:
            rows = 1

        # Calculate block dimensions
        block_width = width // cols
        block_height = height // rows

        image_blocks = []

        # Iterate over rows and columns to split the image into blocks
        for row in range(rows):
            for col in range(cols):
                y_start = row * block_height
                y_end = (row + 1) * block_height
                x_start = col * block_width
                x_end = (col + 1) * block_width

                # Extract each block
                original_block = grayscale_image[
                    y_start:y_end,
                    x_start:x_end,
                ]

                # Resize once for the autoencoder and keep the source block size
                # so the error map can be stitched back into the original frame.
                resized_block = cv2.resize(original_block, (self.procesimgsize, self.procesimgsize))
                block_tensor = torch.from_numpy(resized_block).unsqueeze(0).unsqueeze(0).float() / 255.0

                image_blocks.append(
                    {
                        "tensor": block_tensor,
                        "coords": (y_start, y_end, x_start, x_end),
                        "shape": original_block.shape,
                    }
                )

        return grayscale_image, image_blocks





    def predict_and_calculate_mse(self, image):
        """
        Keep the existing function name to minimize call-site changes, but
        return the full per-pixel squared error map instead of a scalar.
        """
        with torch.no_grad():
            decoded_image = self.autoencoder(image)
            error_map = torch.square(image - decoded_image)

        return error_map

    def predict_activation_heatmap(self, image):
        """
        Build a smoother visualization map from deep encoder activations.
        The error map is compressed to the encoder resolution, combined with
        channel-averaged activations, and then upsampled back to block size.
        """
        with torch.no_grad():
            encoded = self.autoencoder.encoder(image)
            decoded_image = self.autoencoder.decoder(encoded)
            error_map = torch.square(image - decoded_image)

            # The encoder output is low-resolution, which naturally produces
            # larger blob-like regions instead of pixel-level noise.
            activation_map = torch.mean(encoded, dim=1, keepdim=True)
            coarse_error_map = F.interpolate(error_map, size=encoded.shape[-2:], mode="area")
            activation_heatmap = activation_map * coarse_error_map
            activation_heatmap = F.interpolate(
                activation_heatmap,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return error_map, activation_heatmap

    def calculate_scalar_mse(self, error_map):
        """
        Derive the scalar block MSE from the already computed error map.
        """
        return torch.mean(error_map).item()

    def stitch_block_map(self, full_map, block_map, block):
        """
        Resize a block-level map back to the source block size and place it into
        the correct position in the full-frame map.
        """
        map_np = block_map.squeeze().detach().cpu().numpy()
        block_height, block_width = block["shape"]
        resized_map = cv2.resize(map_np, (block_width, block_height))
        y_start, y_end, x_start, x_end = block["coords"]
        full_map[y_start:y_end, x_start:x_end] = resized_map

    def calculate_region_scores(self, full_error_map):
        """
        Convert a full-frame curiosity map into the grid of scalar values used
        by the moving-head logic.
        """
        cols, rows = self.movement_grid
        height, width = full_error_map.shape
        region_width = width // cols
        region_height = height // rows
        region_scores = []

        for row in range(rows):
            for col in range(cols):
                y_start = row * region_height
                y_end = (row + 1) * region_height
                x_start = col * region_width
                x_end = (col + 1) * region_width
                region_map = full_error_map[y_start:y_end, x_start:x_end]
                region_scores.append(float(np.mean(region_map)) * 1000)

        return region_scores

    def build_error_overlay(self, grayscale_image, full_error_map):
        """
        Normalize the stitched error map, colorize it, and blend it with the
        source grayscale frame for real-time preview.
        """
        max_error = float(np.max(full_error_map))
        if max_error > 0:
            normalized_map = np.clip((full_error_map / max_error) * 255.0, 0, 255).astype(np.uint8)
        else:
            normalized_map = np.zeros_like(grayscale_image, dtype=np.uint8)

        heatmap = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
        grayscale_bgr = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(grayscale_bgr, 0.65, heatmap, 0.35, 0.0)

    def smooth_visualization_map(self, current_visualization_map):
        """
        Interpolate blob maps across time so the overlay moves more fluidly
        when curiosity updates slower than the camera preview.
        """
        if self.previous_visualization_map is None:
            self.previous_visualization_map = current_visualization_map.copy()
            return current_visualization_map

        smoothed_map = cv2.addWeighted(
            self.previous_visualization_map,
            1.0 - self.visual_smoothing_alpha,
            current_visualization_map,
            self.visual_smoothing_alpha,
            0.0,
        )
        self.previous_visualization_map = smoothed_map
        return smoothed_map

    def update_model_with_new_image(self, image, epochs=2):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.autoencoder(image)
            loss = self.criterion(output, image)
            loss.backward()
            self.optimizer.step()

    def run_curiosity(self):
        self.new_image = getattr(self.CAM, "frame", None)
        if self.new_image is None:
            sleep(self.pause)
            return

        # Preprocess the image into blocks
        grayscale_image, blocks = self.preprocess_image(self.new_image)

        # Initialize list for MSE values
        mse_values = []
        show_heatmap = self.visualization_mode in ("heatmap", "activation_heatmap") and getattr(self.CAM, "preview", False)
        use_full_score_map = show_heatmap or self.movement_grid != self.split_values
        full_error_map = None
        full_visualization_map = None

        if use_full_score_map:
            # Preallocate the stitched score map so one forward pass can feed
            # both movement decisions and the preview overlay.
            full_error_map = np.zeros(grayscale_image.shape, dtype=np.float32)
        if show_heatmap:
            full_visualization_map = np.zeros(grayscale_image.shape, dtype=np.float32)

        # Predict and calculate MSE for each block
        for block in blocks:
            if self.visualization_mode == "activation_heatmap":
                error_map, visualization_map = self.predict_activation_heatmap(block["tensor"])
            else:
                error_map = self.predict_and_calculate_mse(block["tensor"])
                visualization_map = error_map

            mse_values.append(self.calculate_scalar_mse(error_map) * 1000)  # Scale MSE for readability

            if full_error_map is not None:
                self.stitch_block_map(full_error_map, error_map, block)
            if full_visualization_map is not None:
                self.stitch_block_map(full_visualization_map, visualization_map, block)

        # Generate summary of MSE values
        #mse_summary = " ".join([f"B{i+1}:{mse:.2f}" for i, mse in enumerate(mse_values)])
        #print(mse_summary)

        # Update curiosity state
        self.state_vals = self.calculate_region_scores(full_error_map) if full_error_map is not None and self.movement_grid != self.split_values else mse_values
        self.CAM.curiosity_data = self.state_vals
        if full_visualization_map is not None:
            smoothed_visualization_map = self.smooth_visualization_map(full_visualization_map)
            self.CAM.display_frame = self.build_error_overlay(grayscale_image, smoothed_visualization_map)

        # Update the model with all blocks at the end
        for block in blocks:
            self.update_model_with_new_image(block["tensor"], epochs=1)
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
