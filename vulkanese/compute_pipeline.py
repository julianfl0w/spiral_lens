import ctypes
import os
import sys
import time
import json
import vulkan as vk
import re
from . import buffer
from . import synchronization

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sinode"))
)
import sinode.sinode as sinode

here = os.path.dirname(os.path.abspath(__file__))


def getVulkanesePath():
    return here


# THIS CONTAINS EVERYTHING YOU NEED!
# The Vulkanese Compute Pipeline includes the following componenets
# command buffer
# pipeline,
# All in one. it is self-contained
class ComputePipeline(sinode.Sinode):
    def __init__(self, **kwargs):
        sinode.Sinode.__init__(self, **kwargs)
        # set the defaults here
        self.setDefaults(
            workgroupCount=[1, 1, 1],
            signalSemaphores=[],
            waitSemaphores=[],
            waitStages=[],
        )
        self.descriptorPool = self.fromAbove("descriptorPool")

        # synchronization is owned by the pipeline (command buffer?)

        # if this shader is not signalling other ones,
        # it's either a dead-end or an output
        if not len(self.signalSemaphores):
            self.fence = self.device.getFence()

        push_constant_ranges = vk.VkPushConstantRange(stageFlags=0, offset=0, size=0)

        # The pipeline layout allows the pipeline to access descriptor sets.
        # So we just specify the established descriptor set
        self.vkPipelineLayoutCreateInfo = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            flags=0,
            setLayoutCount=len(self.descriptorPool.descSets),
            pSetLayouts=[d.vkDescriptorSetLayout for d in self.descriptorPool.descSets],
            pushConstantRangeCount=0,
            pPushConstantRanges=[push_constant_ranges],
        )

        self.vkPipelineLayout = vk.vkCreatePipelineLayout(
            device=self.device.vkDevice,
            pCreateInfo=self.vkPipelineLayoutCreateInfo,
            pAllocator=None,
        )

        self.vkComputePipelineCreateInfo = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=self.computeShader.vkPipelineShaderStageCreateInfo,
            layout=self.vkPipelineLayout,
        )

        # Now, we finally create the compute pipeline.
        self.vkPipeline = vk.vkCreateComputePipelines(
            device=self.device.vkDevice,
            pipelineCache=vk.VK_NULL_HANDLE,
            createInfoCount=1,
            pCreateInfos=[self.vkComputePipelineCreateInfo],
            pAllocator=None,
        )[0]

        # Now we shall start recording commands into the newly allocated command buffer.
        self.beginInfo = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            # the buffer is only submitted and used once in this application.
            # flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
            flags=0,
        )

        # wrap it all up into a command buffer
        self.vkCommandBufferAllocateInfo = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.device.vkComputeCommandPool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )

        self.vkCommandBuffer = vk.vkAllocateCommandBuffers(
            device=self.device.vkDevice, pAllocateInfo=self.vkCommandBufferAllocateInfo
        )[0]

        vk.vkBeginCommandBuffer(self.vkCommandBuffer, self.beginInfo)

        # We need to bind a pipeline, AND a descriptor set before we dispatch.
        # The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
        vk.vkCmdBindPipeline(
            self.vkCommandBuffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.vkPipeline
        )

        self.parent.parent.dump()
        vk.vkCmdBindDescriptorSets(
            commandBuffer=self.vkCommandBuffer,
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            layout=self.vkPipelineLayout,
            firstSet=0,
            descriptorSetCount=len(self.descriptorPool.activevkDescriptorSets),
            pDescriptorSets=self.descriptorPool.activevkDescriptorSets,
            dynamicOffsetCount=0,
            pDynamicOffsets=None,
        )

        # Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
        # The number of workgroups is specified in the arguments.
        # If you are already familiar with compute shaders from OpenGL, this should be nothing new to you.
        vk.vkCmdDispatch(
            self.vkCommandBuffer,
            self.workgroupCount[0],
            self.workgroupCount[1],
            self.workgroupCount[2],
        )

        vk.vkEndCommandBuffer(self.vkCommandBuffer)

        if len(self.waitSemaphores):
            pWaitSemaphores = [s.vkSemaphore for s in self.waitSemaphores]
        else:
            pWaitSemaphores = None

        if len(self.signalSemaphores):
            pSignalSemaphores = [s.vkSemaphore for s in self.signalSemaphores]
        else:
            pSignalSemaphores = None

        # Information describing the queue submission
        # Now we shall finally submit the recorded command buffer to a queue.
        self.submitInfo = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[self.vkCommandBuffer],
            waitSemaphoreCount=len(self.waitSemaphores),
            pWaitSemaphores=pWaitSemaphores,
            signalSemaphoreCount=len(self.signalSemaphores),
            pSignalSemaphores=pSignalSemaphores,
            pWaitDstStageMask=self.waitStages,
        )

    # this help if you run the main loop in C/C++
    # just use the Vulkan addresses!
    def getVulkanAddresses(self):
        addrDict = {}
        addrDict["FENCEADDR"] = hex(eval(str(self.fence).split(" ")[-1][:-1]))
        addrDict["DEVADDR"] = str(self.device.vkDevice).split(" ")[-1][:-1]
        addrDict["SUBMITINFOADDR"] = str(ffi.addressof(self.submitInfo)).split(" ")[-1][
            :-1
        ]
        return addrDict

    # the main loop
    def run(self, blocking=True):
        vkFence = None
        if hasattr(self, "fence"):
            vkFence = self.fence.vkFence

        # We submit the command buffer on the queue, at the same time giving a fence.
        vk.vkQueueSubmit(
            queue=self.device.compute_queue,
            submitCount=1,
            pSubmits=self.submitInfo,
            fence=vkFence,
        )

        if hasattr(self, "fence"):
            self.wait()

    def wait(self):
        self.fence.wait()

    def release(self):

        self.device.instance.debug("destroying pipeline")
        vk.vkDestroyPipeline(self.device.vkDevice, self.vkPipeline, None)
        self.device.instance.debug("destroying pipeline layout")
        vk.vkDestroyPipelineLayout(self.device.vkDevice, self.vkPipelineLayout, None)
