# Window name in which image is displayed
pass

from screeninfo import get_monitors
import cv2
import sys
#import pygame
import sys
import numpy as np
from PIL import Image
import mmap


class FullScreenPG:
    def __init__(self):
        pygame.init()
        
        # Get the monitor refresh rate?
        self.display = get_monitors()[0]

        self.infoObject = pygame.display.Info()

        self.screen = pygame.display.set_mode((self.infoObject.current_w, self.infoObject.current_h), pygame.FULLSCREEN)

        # set window name
        pygame.display.set_caption("Image")

    def write(self, image):
        image = image[:, :, :3]  # Discard the alpha channel

        # Convert the image to pygame format
        image = pygame.surfarray.make_surface(image)

        # Draw the image onto the screen
        self.screen.blit(image, (0, 0))

        # Update the display
        pygame.display.flip()

        # event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()

# Usage
# img = cv2.imread('image.jpg') 
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# FullScreen().write(np.rot90(img_rgb))


class FullScreen:
    def __init__(self):

        # Get the monitor refresh rate?
        self.display = get_monitors()[0]

        self.window_name = "Image"

        # Create a fullscreen window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Set the window properties to fullscreen   
        cv2.setWindowProperty(
            self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

    def write(self, image):
        # Displaying the image
        cv2.imshow(self.window_name, image)
        key = int(cv2.waitKey(1))
        return key


class PillowScreen:
    def __init__(self):

        # Get the monitor refresh rate?
        self.display = get_monitors()[0]

        self.width = self.display.width
        self.height = self.display.height
        self.fb_file = open('/dev/fb0', 'r+b')
        self.fb = mmap.mmap(self.fb_file.fileno(), 0)

    def write(self, image):
        # Resize the image to match the screen dimensions
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((self.width, self.height))

        # Convert the PIL image to raw framebuffer format
        fb_image = pil_image.tobytes()

        # Write the raw framebuffer data to the framebuffer
        self.fb.seek(0)
        self.fb.write(fb_image)

    def close(self):
        # Close the framebuffer device
        self.fb.close()
        self.fb_file.close()
