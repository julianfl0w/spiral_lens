import json
import sys
import os

here = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(
    0, os.path.join(here, "..", "..", "sinode")
)
import sinode.sinode as sinode

import vulkan as vk
import numpy as np
import time

class StorageImage(sinode.Sinode):
    def __init__(self, **kwargs):
        sinode.Sinode.__init__(self, **kwargs)
        self.setDefaults(format=vk.VK_FORMAT_R8G8B8A8_UNORM, usage=vk.VK_IMAGE_USAGE_STORAGE_BIT | vk.VK_IMAGE_USAGE_SAMPLED_BIT)

        # Create image
        self.imageCreateInfo = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=self.format,
            extent=vk.VkExtent3D(width=self.width, height=self.height, depth=1),
            mipLevels=1,
            arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=self.usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )


        self.image = vk.vkCreateImage(self.device.vkDevice, self.imageCreateInfo, None)

        # Allocate and bind memory
        memoryRequirements = vk.vkGetImageMemoryRequirements(self.device.vkDevice, self.image)
        memoryTypeIndex = self.device.findMemoryType(memoryRequirements.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

        self.memoryAllocateInfo = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=memoryRequirements.size,
            memoryTypeIndex=memoryTypeIndex
        )

        self.memory = vk.vkAllocateMemory(self.device.vkDevice, self.memoryAllocateInfo, None)
        vk.vkBindImageMemory(self.device.vkDevice, self.image, self.memory, 0)

        # Create image view
        self.imageViewCreateInfo = vk.VkImageViewCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            image=self.image,
            viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
            format=self.format,
            components=vk.VkComponentMapping(r=vk.VK_COMPONENT_SWIZZLE_IDENTITY, g=vk.VK_COMPONENT_SWIZZLE_IDENTITY, b=vk.VK_COMPONENT_SWIZZLE_IDENTITY, a=vk.VK_COMPONENT_SWIZZLE_IDENTITY),
            subresourceRange=vk.VkImageSubresourceRange(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1)
        )

        self.imageView = vk.vkCreateImageView(self.device.vkDevice, self.imageViewCreateInfo, None)

    def release(self):
        vk.vkDestroyImageView(self.device.vkDevice, self.imageView, None)
        vk.vkFreeMemory(self.device.vkDevice, self.memory, None)
        vk.vkDestroyImage(self.device.vkDevice, self.image, None)
