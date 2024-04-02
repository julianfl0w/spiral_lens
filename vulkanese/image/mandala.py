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

# from vulkanese.vulkanese import *


#######################################################


class Mandala(ve.shader.Shader):
    def __init__(
        self,
        WIDTH:3200,  # Size of rendered mandelbrot set.
        HEIGHT=2400,  # Size of renderered mandelbrot set.
        **kwargs
    ):
        self.WIDTH = int(WIDTH)
        self.HEIGHT = int(HEIGHT)

        self.proc_kwargs(**kwargs)

        here = os.path.dirname(os.path.abspath(__file__))
        self.setDefaults(
            memProperties=0
            | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            mandleStride=0.001,
            originX = 0,
            originY = 0,
            name="Mandala",
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            sourceFilename=os.path.join(here, "Mandala.template.comp"),
            constantsDict=dict(
                HEIGHT=self.HEIGHT,
                WIDTH=self.WIDTH,
                WORKGROUP_SIZE=32,  # Workgroup size in compute shader.
            ),
            workgroupCount=[
                math.ceil(float(WIDTH) / 32),
                math.ceil(float(HEIGHT) / 32),
                1,
            ],
        )

        self.imageData = ve.buffer.StorageBuffer(
            device=self.device,
            name="imageData",
            qualifier="writeonly",
            memtype="uint",
            format="VK_FORMAT_R8G8B8A8_UINT",
            shape=[WIDTH * HEIGHT],
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        )

        self.paramsBuffer = ve.buffer.StorageBuffer(
            device=self.device,
            name="paramsBuffer",
            qualifier="readonly",
            memtype="double",
            shape=[16],
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        )

        self.buffers=[self.imageData, self.paramsBuffer]

        ve.shader.Shader.__init__(self, device=self.device)
        self.finalize()

        # print the object hierarchy
        print("Object tree:")
        print(json.dumps(self.device.asDict(), indent=4))

    def getImage(self):
        pa = np.frombuffer(self.imageData.pmap, np.uint8)
        pa = pa.reshape((self.HEIGHT, self.WIDTH, 4))
        # pa = self.imageData.get()
        return pa

    def run(self):
        self.paramsBuffer.set(np.array([self.originX, self.originY, self.mandleStride, 0] + [0]*12))
        ve.shader.Shader.run(self)
        return self.getImage()

def runDemo():
    # device selection and instantiation
    instance_inst = ve.instance.Instance(verbose=True)
    screen = ve.screen.FullScreen()
    print("available Devices:")
    # for i, d in enumerate(instance_inst.getDeviceList()):
    # 	print("    " + str(i) + ": " + d.deviceName)
    print("")

    # choose a device
    print("naively choosing device 0")
    device = instance_inst.getDevice(0)
    mandle = Mandala(
        device=device,
        instance=instance_inst,
        parent=device,
        WIDTH=screen.display.width,
        HEIGHT=screen.display.height,
    )
    
    while 1:
        img = mandle.run()
        key = screen.write(img)
        speed = 20
        if key == ord("q"):
            break
        elif key == -1:
            pass
        elif key == ord("a"):
            mandle.zoom(1/1.05)
        elif key == ord("s"):
            mandle.zoom(1.05)
        elif key == 82: #up
            mandle.pan(0, -speed)
        elif key == 84: #down
            mandle.pan(0, speed)
        elif key == 83: #right
            mandle.pan(speed, 0)
        elif key == 81: #left
            mandle.pan(-speed, 0)
        else:
            print(key)

    # elegantly free all memory
    instance_inst.release()

def createVideo():
    # device selection and instantiation
    instance_inst = ve.instance.Instance(verbose=True)
    screen = ve.screen.FullScreen()

    # choose a device
    print("naively choosing device 0")
    device = instance_inst.getDevice(0)
    mandle = Mandala(
        device=device,
        instance=instance_inst,
        parent=device,
        WIDTH=screen.display.width,
        HEIGHT=screen.display.height,
    )
    
    img = mandle.run()
    
    # Below VideoWriter object will create
    # a frame of above defined The output 
    # is stored in 'filename.avi' file.
    print(img.shape)
    size = [int(img.shape[1]), int(img.shape[0])]
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    result = cv2.VideoWriter('output.avi', 
                            fourcc,
                            30, 
                            size, True)
    
    mandle.setCoords([-458.28127551480645, -257.7060843340382, 0.9555938177273168])
    
    for i in range(500):
        img_RGBA = mandle.run()
        img_RGB = cv2.cvtColor(img_RGBA, cv2.COLOR_RGBA2RGB)
        result.write(img_RGB)
        #result.write(np.zeros([size[1], size[0], 4], dtype=np.uint8))
        #screen.write(img)
        mandle.zoom(1.03)


    # elegantly free all memory
    instance_inst.release()
    result.release()


if __name__ == "__main__":  
    runDemo()
    #createVideo()