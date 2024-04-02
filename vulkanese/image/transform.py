#!/bin/env python
import ctypes
import os
import time
import sys
import numpy as np
import json
import cv2
import math

here = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(here, "..", ".."))
sys.path.append(os.path.join(here, "..", "..", "..", "sinode"))
import sinode.sinode as sinode
import vulkanese as ve
import vulkan as vk

# Assuming sinode, vulkanese, and vulkan libraries are correctly imported and set up


class Transform(ve.shader.Shader):
    def __init__(self, device, WIDTH=3200, HEIGHT=2400, CHANNELS=3, **kwargs):
        self.WIDTH = int(WIDTH)
        self.HEIGHT = int(HEIGHT)
        self.CHANNELS = int(CHANNELS)

        self.device = device  # Make sure the device is passed and stored

        # Set shader properties, including the path to the 'transform.template.comp'
        self.setDefaults(
            memProperties=0
            | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            name="Transform",
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            sourceFilename=os.path.join(
                here, "shaders", "transform.template.comp"
                #here, "shaders", "passthrough.template.comp"
            ),  # Updated to use 'transform.template.comp'
            constantsDict=dict(
                HEIGHT=self.HEIGHT,
                WIDTH=self.WIDTH,
                WORKGROUP_SIZE=32,  # Workgroup size in compute shader
            ),
            workgroupCount=[
                math.ceil(float(WIDTH) / 32),
                math.ceil(float(HEIGHT) / 32),
                1,
            ],
        )

        # Input image setup (assuming it's loaded or captured elsewhere)
        self.inputImage = ve.buffer.StorageBuffer(
            device=self.device,
            name="inputImage",
            memtype="vec4",
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            shape=[WIDTH, HEIGHT, CHANNELS],
        )

        # Output image setup
        self.outputImage = ve.buffer.StorageBuffer(
            device=self.device,
            name="outputImage",
            memtype="vec4", 
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            shape=[WIDTH, HEIGHT, CHANNELS],
        )

        # Transformation parameters buffer setup
        self.paramsBuffer = ve.buffer.UniformBuffer(
            device=self.device,
            name="paramsBuffer",
            memtype="float",  # Assuming your transformation parameters are floats
            shape=[128],  # Scale, rotation angle, translation (x, y)
            params={
                "width": "float",
                "height": "float",
                "scale": "float",
                "rotationAngle": "float",
                "translationX": "float",
                "translationY": "float",
            },
        )

        self.buffers = [self.paramsBuffer, self.inputImage, self.outputImage]

        # Initialize the base shader
        ve.shader.Shader.__init__(self, device=self.device)
        self.finalize()

    def run(self, input_img, scale, rotation_angle, translation):
        # Set transformation parameters
        self.paramsBuffer.setByIndex(
            0,
            np.array(
                [
                    self.WIDTH,
                    self.HEIGHT,
                    scale,
                    rotation_angle,
                    translation[0],
                    translation[1],
                ],
                dtype=np.float32,
            ),
        )

        # Upload the input image to GPU
        self.inputImage.set(input_img)

        # Execute the shader
        ve.shader.Shader.run(self)

        # Retrieve and return the transformed image
        return self.outputImage.get()


# Example usage
def runDemo():

    # device selection and instantiation
    instance_inst = ve.instance.Instance(verbose=True)
    instance_inst.DEBUG = True
    device = instance_inst.getDevice(0)

    # Load or capture an input image...
    input_img = cv2.imread("example.png")  # Example input image
    input_img = input_img.astype(np.float32)/255.0

    # Create a Transform object
    transform = Transform(
        device=device, WIDTH=input_img.shape[0], HEIGHT=input_img.shape[1]
    )

    # Run the transformation (example parameters)
    output_img = transform.run(
        #input_img, scale=1.0, rotation_angle=math.pi / 4, translation=[0.1, 0.1]
        input_img, scale=1.0, rotation_angle=0, translation=[0,0]
    )
    print(output_img.shape)
    print(output_img.dtype)
    print(output_img.flatten()[:16])
    cv2.imshow('Output Image', output_img)
    cv2.waitKey(0)


if __name__ == "__main__":
    runDemo()
