import json
from sinode import *
import os

here = os.path.dirname(os.path.abspath(__file__))
from vulkan import *


class DescriptorPool(Sinode):
    def __init__(self, device, MAX_FRAMES_IN_FLIGHT=3):
        Sinode.__init__(self, device)
        self.device = device
        self.vkDevice = device.vkDevice
        self.MAX_FRAMES_IN_FLIGHT = MAX_FRAMES_IN_FLIGHT

        # The descriptor set number 0 will be used for engine-global resources, and bound once per frame.
        self.descSetGlobal = DescriptorSet(
            self, binding=0, name="global", type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        )
        # The descriptor set number 2 will be used for material resources,
        self.descSetUniform = DescriptorSet(
            self, binding=1, name="uniform", type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        )

        # The descriptor set number 1 will be used for per-pass resources, and bound once per pass.
        self.descSetPerPass = DescriptorSet(
            self, binding=2, name="perPass", type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        )

        # and the number 3 will be used for per-object resources.
        self.descSetPerObject = DescriptorSet(
            self, binding=3, name="perObject", type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        )

        self.descSets = [
            self.descSetGlobal,
            self.descSetUniform,
            self.descSetPerPass,
            self.descSetPerObject,
        ]

    def getBinding(self, buffer, descSet):
        self.device.instance.debug("Allocating ")
        self.device.instance.debug(buffer)
        self.device.instance.debug(bindname)
        return self.descSet.attachBuffer(buffer)

    # We first need to describe which descriptor types our descriptor sets are going to contain and how many of them, using VkDescriptorPoolSize structures.
    def finalize(self):
        # create descriptor pool.
        # self.poolSize = VkDescriptorPoolSize(
        #    type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=4
        # )

        # 2 uniform pools, 2 storage?
        self.poolSizeS = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=1000,  # len(self.descSets[0].buffers)
        )
        self.poolSizeU = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1000,  # len(self.descSets[1].buffers)
        )
        
        # We will allocate one of these descriptors for every frame. This pool size structure is referenced by the main VkDescriptorPoolCreateInfo:
        # Aside from the maximum number of individual descriptors that are available, we also need to specify the maximum number of descriptor sets that may be allocated:
        self.poolInfo = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=2,
            pPoolSizes=[self.poolSizeS, self.poolSizeU],
            maxSets=4,
        )  # imposed by some gpus

        # The structure has an optional flag similar to command pools that determines if individual descriptor sets can be freed or not: VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT. We're not going to touch the descriptor set after creating it, so we don't need this flag. You can leave flags to its default value of 0.
        self.vkDescriptorPool = vkCreateDescriptorPool(
            device=self.vkDevice, pCreateInfo=[self.poolInfo], pAllocator=0
        )

        # This way, the inner render loops will only be binding descriptor sets 2 and 3, and performance will be high.
        for descriptor in self.descSets:
            descriptor.finalize()

        # Establish the create info
        descriptorSetAllocateInfo = VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.vkDescriptorPool,
            descriptorSetCount=len(self.descSets),
            pSetLayouts=[d.vkDescriptorSetLayout for d in self.descSets],
        )

        # create the allocate descriptor set.
        self.vkDescriptorSets = vkAllocateDescriptorSets(
            self.vkDevice, descriptorSetAllocateInfo
        )

        self.writeDescriptorSets = []
        self.activevkDescriptorSets = []
        self.activeDescriptorSets = []
        self.device.instance.debug(self.vkDescriptorSets)
        for i, d in enumerate(self.descSets):

            # The Vulkan spec states: descriptorCount must be greater than 0
            if not len(d.buffers):
                continue
            d.vkDescriptorSet = self.vkDescriptorSets[i]

            # Next, we need to connect our actual storage buffer with the descrptor.
            # We use vkUpdateDescriptorSets() to update the descriptor set.

            # descCOUNT = max([b.binding for b in d.buffers])+1
            # one descriptor per buffer?
            d.vkWriteDescriptorSet = VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=d.vkDescriptorSet,
                # dstBinding=d.binding,
                dstBinding=0,
                descriptorCount=len(d.buffers),
                descriptorType=d.type,
                pBufferInfo=[b.descriptorBufferInfo for b in d.buffers],
            )
            self.device.instance.debug("Buffers")
            self.device.instance.debug(d.buffers)
            self.device.instance.debug("buffbinds")
            self.device.instance.debug([b.binding for b in d.buffers])
            self.device.instance.debug("binding")
            self.device.instance.debug(d.binding)
            self.device.instance.debug("type")
            self.device.instance.debug(d.type)

            self.writeDescriptorSets += [d.vkWriteDescriptorSet]
            self.activevkDescriptorSets += [d.vkDescriptorSet]
            self.activeDescriptorSets += [d]

        self.device.instance.debug(self.writeDescriptorSets)
        for i, wDescSet in enumerate(self.writeDescriptorSets):
            if i == 0:
                self.device.instance.debug("WRITING GLOBAL DESC SET")
            else:
                self.device.instance.debug("WRITING UNIFORM DESC SET")
            # perform the update of the descriptor set.
            vkUpdateDescriptorSets(
                device=self.vkDevice,
                descriptorWriteCount=1,  # len(writeDescriptorSets),
                pDescriptorWrites=wDescSet,
                descriptorCopyCount=0,
                pDescriptorCopies=None,
            )

    def release(self):
        for v in self.descSets:
            self.device.instance.debug("destroying descriptor set " + v.name)
            v.release()
        self.device.instance.debug("destroying descriptor pool")
        vkDestroyDescriptorPool(self.vkDevice, self.vkDescriptorPool, None)


class DescriptorSet(Sinode):
    def __init__(self, descriptorPool, binding, name, type, MAX_FRAMES_IN_FLIGHT=3):
        Sinode.__init__(self, descriptorPool)
        self.name = name
        self.device = descriptorPool.device
        self.vkDevice = descriptorPool.vkDevice
        self.descriptorPool = descriptorPool
        self.buffers = []
        self.type = type
        self.binding = binding
        self.currBufferBinding = 0

    def getBufferBinding(self):
        retval = self.currBufferBinding
        self.currBufferBinding += 1
        return retval

    def finalize(self):
        # Here we specify a descriptor set layout. This allows us to bind our descriptors to
        # resources in the shader.
        # Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to the binding point
        # 0. This binds to
        #   layout(std140, binding = 0) buffer buf
        # in the compute shader.

        # Establish the create info
        descriptorSetLayoutCreateInfo = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(self.buffers),
            pBindings=[buffer.descriptorSetLayoutBinding for buffer in self.buffers],
        )

        # Create the descriptor set layout.
        self.vkDescriptorSetLayout = vkCreateDescriptorSetLayout(
            self.vkDevice, descriptorSetLayoutCreateInfo, None
        )
        self.device.instance.debug("finalized desc set " + self.name)

    def release(self):
        for b in self.buffers:
            b.release()
        self.device.instance.debug("destroying descriptor set")
        vkDestroyDescriptorSetLayout(self.vkDevice, self.vkDescriptorSetLayout, None)


# 	def updateDescriptorSet():
# 		std::vector<VkWriteDescriptorSet> writes;
#
# 		# Camera matrices and scene description
# 		VkDescriptorBufferInfo dbiUnif{self.bGlobals.buffer, 0, VK_WHOLE_SIZE};
# 		writes += [self.descSetLayoutBind.makeWrite(self.descSet, "engineGlobal", &dbiUnif));
#
# 		VkDescriptorBufferInfo dbiSceneDesc{self.bObjDesc.buffer, 0, VK_WHOLE_SIZE};
# 		writes += [self.descSetLayoutBind.makeWrite(self.descSet, "perObject", &dbiSceneDesc));
#
# 		# All texture samplers
# 		std::vector<VkDescriptorImageInfo> diit;
# 		for(auto& texture : self.textures)
# 		{
# 		diit += [texture.descriptor);
# 		}
# 		writes += [self.descSetLayoutBind.makeWriteArray(self.descSet, "perPass", diit.data()));
#
# 		# Writing the information
# 		vkUpdateDescriptorSets(self.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

# 	def updatePostDescriptorSet():
# 		VkWriteDescriptorSet writeDescriptorSets = self.postDescSetLayoutBind.makeWrite(self.postDescSet, 0, &self.offscreenColor.descriptor);
# 		vkUpdateDescriptorSets(self.device, 1, &writeDescriptorSets, 0, nullptr);

# 	def createPostDescriptor():
# 		self.postDescSetLayoutBind.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
# 		self.postDescSetLayout = self.postDescSetLayoutBind.createLayout(self.device);
# 		self.postDescPool      = self.postDescSetLayoutBind.createPool(self.device);
# 		self.postDescSet       = nvvk::allocateDescriptorSet(self.device, self.postDescPool, self.postDescSetLayout)
