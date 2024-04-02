#!/bin/env python
import ctypes
import os
import time
import sys
import numpy as np
import json
import cv2
import math
import json

here = os.path.dirname(os.path.abspath(__file__))

import requests

from dotenv import load_dotenv

sys.path.append(os.path.join(here, "..", ".."))
sys.path.append(os.path.join(here, "..", "..", "..", "sinode"))
import sinode.sinode as sinode
import vulkanese as ve
import vulkan as vk


class ShaderToy(ve.graphics_pipeline.GraphicsPipeline):
    def __init__(self, **kwargs):
        self.proc_kwargs(**kwargs)
        self.WIDTH = int(self.WIDTH / 2)
        self.HEIGHT = int(self.HEIGHT / 2)
        print(self.shaderToyAPIKey)
        url = f"https://www.shadertoy.com/api/v1/shaders/{self.shaderID}?key={self.shaderToyAPIKey}"
        print(url)
        # response = requests.post(url)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.post(url, headers=headers)

        print(response)
        json_data = json.loads(response.text)
        print(json.dumps(json_data, indent=2))
        code = json_data["Shader"]["renderpass"][0]["code"]
        code = code.encode().decode("unicode_escape")
        # print(code)
        code = (
            """
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_gpu_shader_fp64 : enable

#define gl_FragCoord gl_GlobalInvocationID
DEFINE_STRING// This will be (or has been) replaced by constant definitions
BUFFERS_STRING// This will be (or has been) replaced by buffer definitions
    
layout (local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE, local_size_z = 1 ) in;

        """
            + code
            + """

void main() {

    vec4 color;
    vec2 fragCoord = vec2(float(gl_GlobalInvocationID.x), HEIGHT - 1 - float(gl_GlobalInvocationID.y));
    mainImage(color, fragCoord);

    // Convert the color components to 8-bit unsigned integer values
    uint c_r = uint(color.r * 255.0);
    uint c_g = uint(color.g * 255.0);
    uint c_b = uint(color.b * 255.0);
    uint c_a = uint(color.a * 255.0);


    // Pack the components into a single uint value
    uint pixelValue = (c_a << 24) | (c_b << 16) | (c_g << 8) | c_r;

    // store the rendered mandelbrot set into a storage buffer:
    imageData[WIDTH * gl_GlobalInvocationID.y + gl_GlobalInvocationID.x] = pixelValue;
}
        """
        )

        self.setDefaults(
            shaderID="ldlXRS",
            shaderToyAPIKey="none",
            memProperties=0
            | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            shaderToyStride=0.001,
            originX=0,
            originY=0,
            name="ShaderToy",
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            fragSourceText=code,
            HEIGHT=self.HEIGHT,
            WIDTH=self.WIDTH,
            WORKGROUP_SIZE=32,  # Workgroup size in compute shader.
            workgroupCount=[
                math.ceil(float(self.WIDTH) / 32),
                math.ceil(float(self.HEIGHT) / 32),
                1,
            ],
        )

        self.vertexBuffers = [
            ve.buffer.StorageBuffer(
                device=self.device,
                name="fragBuffer",
                qualifier="out",
                memtype="vec4",
                format="VK_FORMAT_R8G8B8A8_UINT",
                shape=[self.WIDTH * self.HEIGHT],
                stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
                location=2,
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="position",
                qualifier="in",
                memtype="vec3",
                shape=[12],
                location=1,
                stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="color",
                qualifier="in",
                memtype="vec4",
                shape=[12],
                location=0,
                stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
            ),
        ]
        self.fragBuffers = [
            ve.buffer.StorageBuffer(
                device=self.device,
                name="imageData",
                qualifier="writeonly",
                memtype="uint",
                format="VK_FORMAT_R8G8B8A8_UINT",
                shape=[self.WIDTH * self.HEIGHT],
                stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="iTime",
                qualifier="readonly",
                memtype="float",
                stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="iResolution",
                qualifier="readonly",
                memtype="vec4",
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="iMouse",
                qualifier="readonly",
                memtype="vec4",
                stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            ),
        ]

        #self.finalize()
        self.shaders = [
            # (vertex -> tesselate -> fragment)
            # Vertex Stage
            ve.shader.VertexStage(
                device=self.device,
                buffers=self.vertexBuffers,
                name="vertexStage",
                sourceFilename=os.path.join(here, "shaders", "passthrough.template.vert"),
                constantsDict={}
            )]
        
        self.gpuBuffers = sinode.Generic()
        for b in self.vertexBuffers + self.fragBuffers:
            exec(f"self.gpuBuffers.{b.name}=b")

        self.gpuBuffers.fragBuffer.qualifier = "in"
        self.shaders += [
            # fragment stage
            ve.shader.FragmentStage(
                device=self.device,
                buffers=self.fragBuffers,
                name="fragmentStage",
                sourceFilename=os.path.join(here, "shaders", "passthrough.template.frag"),
                constantsDict={}
            )
        ]

        ve.graphics_pipeline.GraphicsPipeline.__init__(self)

        self.gpuBuffers.iResolution.set(np.array([self.WIDTH, self.HEIGHT, 0, 0]))
        # print the object hierarchy
        print("Object tree:")
        print(json.dumps(self.device.asDict(), indent=4))
        self.startTime = time.time()

    def getImage(self):
        pa = np.frombuffer(self.gpuBuffers.imageData.pmap, np.uint8)
        pa = pa.reshape((self.HEIGHT, self.WIDTH, 4))
        # pa = self.imageData.get()
        return pa

    def dumpPos(self):
        print([self.originX, self.originY, self.shaderToyStride])

    def setCoords(self, coords):
        self.originX, self.originY, self.shaderToyStride = coords

    def zoom(self, amount):
        shaderToyWidth = self.shaderToyStride * self.WIDTH
        shaderToyHeight = self.shaderToyStride * self.HEIGHT
        self.shaderToyStride /= amount
        newshaderToyWidth = self.shaderToyStride * self.WIDTH
        newshaderToyHeight = self.shaderToyStride * self.HEIGHT

        self.originX += (shaderToyWidth - newshaderToyWidth) / 2
        self.originY += (shaderToyHeight - newshaderToyHeight) / 2

        self.dumpPos()

    def pan(self, x, y):
        self.originX += x * self.shaderToyStride
        self.originY += y * self.shaderToyStride
        self.dumpPos()

    def run(self):
        # self.paramsBuffer.set(np.array([self.originX, self.originY, self.shaderToyStride, 0] + [0]*12))
        self.gpuBuffers.iTime.set(np.array(time.time() - self.startTime))
        ve.shader.Shader.run(self)
        return self.getImage()


if __name__ == "__main__":
    # device selection and instantiation
    instance_inst = ve.instance.Instance(verbose=True)
    # screen = ve.screen.FullScreenPG()
    screen = ve.screen.FullScreen()

    load_dotenv(os.path.join(here, ".env"))

    # choose a device
    print("naively choosing device 0")
    device = instance_inst.getDevice(0)
    shaderToy = ShaderToy(
        device=device,
        #shaderID="Wt33Wf",
        #shaderID="3dX3zj",
        shaderID="4t33D2",
        shaderToyAPIKey=os.getenv("shadertoy_appkey"),
        instance=instance_inst,
        parent=device,
        WIDTH=screen.display.width,
        HEIGHT=screen.display.height,
    )

    while 1:
        img = shaderToy.run()
        key = screen.write(img)
        speed = 20
        if key == ord("q"):
            break
        elif key == -1:
            pass
        elif key == ord("a"):
            shaderToy.zoom(1 / 1.1)
        elif key == ord("s"):
            shaderToy.zoom(1.1)
        elif key == 82:  # up
            shaderToy.pan(0, -speed)
        elif key == 84:  # down
            shaderToy.pan(0, speed)
        elif key == 83:  # right
            shaderToy.pan(speed, 0)
        elif key == 81:  # left
            shaderToy.pan(-speed, 0)
        else:
            print(key)

    # elegantly free all memory
    instance_inst.release()
