import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, "..", "..", "sinode")))
import sinode.sinode as sinode

sys.path.insert(0, os.path.abspath(os.path.join(here, "..")))
import vulkanese as ve
import vulkan as vk


class DescriptorSet(sinode.Sinode):
    def __init__(self, descriptorPool, binding, name, type, MAX_FRAMES_IN_FLIGHT=3):
        sinode.Sinode.__init__(self, parent=descriptorPool)
        self.name = name
        self.device = descriptorPool.device
        self.vkDevice = descriptorPool.vkDevice
        self.descriptorPool = descriptorPool
        self.buffers = []
        self.type = type
        self.binding = binding
        self.currBufferBinding = 0

    def bind(self, buffer):
        self.debug("Allocating ")
        self.debug(buffer)
        return self.descSet.attachBuffer(buffer)

    def getBindingNumber(self, buffer):
        self.debug("Returning binding number ")
        self.debug(buffer)
        return self.buffers.index(buffer)

    def addBuffer(self, newBuffer):
        newBuffer.binding = self.currBufferBinding
        self.currBufferBinding += 1

        newBuffer.descriptorSetBinding = self.binding

        self.buffers += [newBuffer]
        self.children += [newBuffer]
        self.parent.buffers += [newBuffer]
        # descriptorCount is the number of descriptors contained in the binding,
        # accessed in a shader as an array, except if descriptorType is
        # VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK in which case descriptorCount
        # is the size in bytes of the inline uniform block
        # self.device.instance.debug("BUFFER DTYPE")
        # self.device.instance.debug(descriptorSet.type)
        newBuffer.descriptorSetLayoutBinding = vk.VkDescriptorSetLayoutBinding(
            binding=newBuffer.binding,
            descriptorType=self.type,
            descriptorCount=1,
            stageFlags=newBuffer.stageFlags,
        )

        # Specify the buffer to bind to the descriptor.
        # Every buffer contains its own info for descriptor set
        # Next, we need to connect our actual storage buffer with the descrptor.
        # We use vkUpdateDescriptorSets() to update the descriptor set.
        newBuffer.descriptorBufferInfo = vk.VkDescriptorBufferInfo(
            buffer=newBuffer.vkBuffer, offset=0, range=newBuffer.sizeBytes
        )

        # if "descriptorSet" not in kwargs.keys():
        #    self.descriptorSet = self.fromAbove("descriptorPool").descSetGlobal

    def getComputeDeclaration(self):
        BUFFERS_STRING = ""
        # novel INPUT buffers belong to THIS Stage (others are linked)

        for buffer in self.buffers:
            if (
                self.fromAbove("stage") == vk.VK_SHADER_STAGE_FRAGMENT_BIT
                and b.name == "fragColor"
            ):
                buffer.qualifier = "in"

            if self.fromAbove("stage") != vk.VK_SHADER_STAGE_COMPUTE_BIT:
                BUFFERS_STRING += b.getDeclaration(descSet=self.descriptorSet)
            else:
                if buffer.usage == vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT:
                    b = "uniform "
                    std = "std140"
                else:
                    b = "buffer "
                    if buffer.compress:
                        std = "std430"
                    else:
                        std = "std140"

                if len(buffer.params):
                    param_definitions = "\n".join([f"{v} {k};" for k, v in buffer.params.items()])
                    BUFFERS_STRING += (
                        f"layout({std}, set = {self.binding}, binding = {self.getBindingNumber(buffer)} ) "
                        + f"{b}{buffer.name}_buf\n" + "{\n   "
                        + param_definitions
                        + "\n};\n"
                    )
                else:
                    BUFFERS_STRING += (
                        f"layout({std}, set = {self.binding}, binding = {self.getBindingNumber(buffer)} ) "
                        + f"{b}{buffer.name}_buf\n" + "{\n   "
                        + buffer.qualifier
                        + " "
                        + buffer.memtype
                        + " "
                        + buffer.name
                        + "["
                        + str(int(buffer.sizeBytes / buffer.itemSizeBytes))
                        + "];\n};\n"
                    )

        if hasattr(self, "DEBUG") and self.DEBUG:
            BUFFERS_STRING = self.addIndicesToOutputs(BUFFERS_STRING)

        return BUFFERS_STRING

    def finalize(self):
        # Here we specify a descriptor set layout. This allows us to bind our descriptors to
        # resources in the shader.
        # Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to the binding point
        # 0. This binds to
        #   layout(std140, binding = 0) buffer buf
        # in the compute shader.

        # Establish the create info
        descriptorSetLayoutCreateInfo = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(self.buffers),
            pBindings=[buffer.descriptorSetLayoutBinding for buffer in self.buffers],
        )

        # Create the descriptor set layout.
        self.vkDescriptorSetLayout = vk.vkCreateDescriptorSetLayout(
            self.vkDevice, descriptorSetLayoutCreateInfo, None
        )
        self.device.instance.debug(
            "finalized desc set "
            + self.name
            + " with "
            + str(len(self.buffers))
            + " buffers"
        )

    def release(self):
        for child in self.children:
            child.release()
        self.device.instance.debug("destroying descriptor set layout")
        if hasattr(self, "vkDescriptorSetLayout"):
            vk.vkDestroyDescriptorSetLayout(
                self.vkDevice, self.vkDescriptorSetLayout, None
            )
