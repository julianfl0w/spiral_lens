import json
from vutil import *
import os

from vulkan import *
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))


class Buffer(Sinode):

    # find memory type with desired properties.
    def findMemoryType(self, memoryTypeBits, properties):
        memoryProperties = vkGetPhysicalDeviceMemoryProperties(
            self.device.physical_device
        )

        # How does this search work?
        # See the documentation of VkPhysicalDeviceMemoryProperties for a detailed description.
        for i, mt in enumerate(memoryProperties.memoryTypes):
            if (
                memoryTypeBits & (1 << i)
                and (mt.propertyFlags & properties) == properties
            ):
                return i

        return -1

    def __init__(
        self,
        device,
        name,
        location,
        descriptorSet,
        format,
        binding = 0,
        usage=VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        memProperties=VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        SIZEBYTES=65536,
        stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
        qualifier="in",
        type="vec3",
    ):
        self.binding = binding
        # this should be fixed in vulkan wrapper
        self.released = False
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT = 0x00020000
        self.usage = usage
        Sinode.__init__(self, device)
        self.device = device
        self.location = location
        self.vkDevice = device.vkDevice
        self.size = SIZEBYTES
        self.qualifier = qualifier
        self.type = type
        self.name = name
        self.descriptorSet = descriptorSet

        print("creating buffer " + name)

        # We will now create a buffer with these options
        self.bufferCreateInfo = VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=SIZEBYTES,  # buffer size in bytes.
            usage=usage,  # buffer is used as a storage buffer.
            sharingMode=sharingMode,  # buffer is exclusive to a single queue family at a time.
        )
        print(self.vkDevice)
        print(self.bufferCreateInfo)

        self.vkBuffer = vkCreateBuffer(self.vkDevice, self.bufferCreateInfo, None)
        self.children += [self.vkBuffer]

        # But the buffer doesn't allocate memory for itself, so we must do that manually.

        # First, we find the memory requirements for the buffer.
        memoryRequirements = vkGetBufferMemoryRequirements(self.vkDevice, self.vkBuffer)

        # There are several types of memory that can be allocated, and we must choose a memory type that:
        # 1) Satisfies the memory requirements(memoryRequirements.memoryTypeBits).
        # 2) Satifies our own usage requirements. We want to be able to read the buffer memory from the GPU to the CPU
        #    with vkMapMemory, so we set VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.
        # Also, by setting VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory written by the device(GPU) will be easily
        # visible to the host(CPU), without having to call any extra flushing commands. So mainly for convenience, we set
        # this flag.
        index = self.findMemoryType(memoryRequirements.memoryTypeBits, memProperties)
        # Now use obtained memory requirements info to allocate the memory for the buffer.
        self.allocateInfo = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=memoryRequirements.size,  # specify required memory.
            memoryTypeIndex=index,
        )

        # allocate memory on device.
        self.vkDeviceMemory = vkAllocateMemory(self.vkDevice, self.allocateInfo, None)
        self.children += [self.vkDeviceMemory]

        # Now associate that allocated memory with the buffer. With that, the buffer is backed by actual memory.
        vkBindBufferMemory(
            device = self.vkDevice, 
            buffer = self.vkBuffer, 
            memory = self.vkDeviceMemory, 
            memoryOffset = 0)

        # Map the buffer memory, so that we can read from it on the CPU.
        self.pmap = vkMapMemory(
            device = self.vkDevice,
            memory = self.vkDeviceMemory, 
            offset = 0, 
            size   = SIZEBYTES, 
            flags  = 0)

        self.bufferDeviceAddressInfo = VkBufferDeviceAddressInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            pNext=None,
            buffer=self.vkBuffer,
        )

        self.descriptorSetLayoutBinding = VkDescriptorSetLayoutBinding(
            binding=self.binding,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=1,
            stageFlags=stageFlags,
        )

        descriptorSet.buffers += [self]
        
        # Specify the buffer to bind to the descriptor.
        self.descriptorBufferInfo = VkDescriptorBufferInfo(
            buffer=self.vkBuffer, offset=0, range=self.size
        )
            
        print("finished creating buffer")

    def saveAsImage(self, height, width, path="mandelbrot.png"):

        # Get the color data from the buffer, and cast it to bytes.
        # We save the data to a vector.
        st = time.time()

        pa = np.frombuffer(self.pmap, np.float32)
        pa = pa.reshape((height, width, 4))
        pa *= 255

        self.cpuDataConverTime = time.time() - st

        # Done reading, so unmap.
        # vkUnmapMemory(self.vkDevice, self.__bufferMemory)

        # Now we save the acquired color data to a .png.
        image = pilImage.fromarray(pa.astype(np.uint8))
        image.save(path)

    def release(self):
        if not self.released:
            print("destroying buffer " + self.name)
            vkFreeMemory(self.vkDevice, self.vkDeviceMemory, None)
            vkDestroyBuffer(self.vkDevice, self.vkBuffer, None)
            self.released = True

    def getDeclaration(self):
        return (
            "layout (location = "
            + str(self.location)
            + ") "
            + self.qualifier
            + " "
            + self.type
            + " "
            + self.name
            + ";\n"
        )

    def getComputeDeclaration(self):
        return (
            "layout(std140, binding = "
            + str(self.binding)
            #+ ", "
            #+ "xfb_stride = " + str(self.stride)
            + ") buffer " + self.name + "_buf\n{\n   "
            #+ self.qualifier
            #+ " "
            + self.type
            + " "
            + self.name
            + "[];\n};\n"
        )
    
    def setBuffer(self, data):
        self.pmap[: data.size * data.itemsize] = data

    def getSize(self):
        with open(os.path.join(here, "derivedtypes.json"), "r") as f:
            derivedDict = json.loads(f.read())
        with open(os.path.join(here, "ctypes.json"), "r") as f:
            cDict = json.loads(f.read())
        size = 0
        if self.type in derivedDict.keys():
            for subtype in derivedDict[self.type]:
                size += self.getSize(subtype)
        else:
            size += 1
        return int(size)


class VertexBuffer(Buffer):
    def __init__(
        self,
        device,
        name,
        location,
        descriptorSet,
        binding = 0,
        format =VK_FORMAT_R32G32B32_SFLOAT,
        usage=VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        rate=VK_VERTEX_INPUT_RATE_VERTEX,
        memProperties=VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        SIZEBYTES=65536,
        qualifier="in",
        type="vec3",
        stride=12,
    ):

        Buffer.__init__(
            self=self,
            location=location,
            binding = binding,
            device=device,
            name=name,
            usage=usage,
            descriptorSet=descriptorSet,
            memProperties=memProperties,
            sharingMode=sharingMode,
            SIZEBYTES=SIZEBYTES,
            qualifier=qualifier,
            type=type,
            format=format,
        )

        outfilename = os.path.join(here, "resources", "standard_bindings.json")
        with open(outfilename, "r") as f:
            bindDict = json.loads(f.read())


        # we will standardize its bindings with a attribute description
        self.attributeDescription = VkVertexInputAttributeDescription(
            binding=self.binding,
            location=self.location,
            format=format,
            offset=0,
        )
        # ^^ Consider VK_FORMAT_R32G32B32A32_SFLOAT  ?? ^^
        self.bindingDescription = VkVertexInputBindingDescription(
            binding=self.binding, stride=stride, inputRate=rate  # 4 bytes/element
        )

        # Every buffer contains its own info for descriptor set
        # Next, we need to connect our actual storage buffer with the descrptor.
        # We use vkUpdateDescriptorSets() to update the descriptor set.
        self.descriptorBufferInfo = VkDescriptorBufferInfo(
            buffer=self.vkBuffer, offset=0, range=SIZEBYTES
        )

        # VK_VERTEX_INPUT_RATE_VERTEX: Move to the next data entry after each vertex
        # VK_VERTEX_INPUT_RATE_INSTANCE: Move to the next data entry after each instance

    def getDeclaration(self):
        if "uniform" in self.qualifier:
            return (
                "layout (location = "
                + str(self.location)
                + ", binding = "
                + str(self.binding)
                + ") "
                + self.qualifier
                + " "
                + self.type
                + " "
                + self.name
                + ";\n"
            )
        else:
            return (
                "layout (location = "
                + str(self.location)
                + ") "
                + self.qualifier
                + " "
                + self.type
                + " "
                + self.name
                + ";\n"
            )


class DescriptorSetBuffer(Buffer):
    def __init__(self, device, setupDict):
        Buffer.__init__(self, device, setupDict)


class PushConstantsBuffer(DescriptorSetBuffer):
    def __init__(self, device, setupDict):
        DescriptorSetBuffer.__init__(self, device, setupDict)


class UniformBuffer(DescriptorSetBuffer):
    def __init__(self, device, setupDict):
        DescriptorSetBuffer.__init__(self, device, setupDict)


class UniformTexelBuffer(DescriptorSetBuffer):
    def __init__(self, device, setupDict):
        DescriptorSetBuffer.__init__(self, device, setupDict)


class SampledImageBuffer(DescriptorSetBuffer):
    def __init__(self, device, setupDict):
        DescriptorSetBuffer.__init__(self, device, setupDict)


class StorageBuffer(DescriptorSetBuffer):
    def __init__(self, device, setupDict):
        DescriptorSetBuffer.__init__(self, device, setupDict)


class StorageTexelBuffer(DescriptorSetBuffer):
    def __init__(self, device, setupDict):
        DescriptorSetBuffer.__init__(self, device, setupDict)


class StorageImageBuffer(DescriptorSetBuffer):
    def __init__(self, device, setupDict):
        DescriptorSetBuffer.__init__(self, device, setupDict)


class AccelerationStructure(DescriptorSetBuffer):
    def __init__(self, setupDict, shader):
        DescriptorSetBuffer.__init__(self, shader)
        self.pipeline = shader.pipeline
        self.pipelineDict = self.pipeline.setupDict
        self.vkCommandPool = self.pipeline.device.vkCommandPool
        self.device = self.pipeline.device
        self.vkDevice = self.pipeline.device.vkDevice
        self.outputWidthPixels = self.pipeline.outputWidthPixels
        self.outputHeightPixels = self.pipeline.outputHeightPixels


class AccelerationStructureNV(AccelerationStructure):
    def __init__(self, setupDict, shader):
        AccelerationStructure.__init__(self, setupDict, shader)

        # We need to get the compactedSize with a query

        # // Get the size result back
        # std::vector<VkDeviceSize> compactSizes(m_blas.size());
        # vkGetQueryPoolResults(m_device, queryPool, 0, (uint32_t)compactSizes.size(), compactSizes.size() * sizeof(VkDeviceSize),
        # 											compactSizes.data(), sizeof(VkDeviceSize), VK_QUERY_RESULT_WAIT_BIT);

        # just playing. we will guess that b***h

        # Provided by VK_NV_ray_tracing
        self.asCreateInfo = VkAccelerationStructureCreateInfoNV(
            sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV,
            pNext=None,
            compactedSize=642000,  # VkDeviceSize
        )

        # Provided by VK_NV_ray_tracing
        self.vkAccelerationStructure = vkCreateAccelerationStructureNV(
            device=self.vkDevice, pCreateInfo=self.asCreateInfo, pAllocator=None
        )


# If type is VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV then geometryCount must be 0
class TLASNV(AccelerationStructureNV):
    def __init__(self, setupDict, shader):
        AccelerationStructureNV.__init__(self, setupDict, shader)

        for blasName, blasDict in setupDict["blas"].items():
            newBlas = BLASNV(blasDict, shader)
            self.children += [newBlas]

        # Provided by VK_NV_ray_tracing
        self.asInfo = VkAccelerationStructureInfoNV(
            sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV,
            pNext=None,  # const void*
            type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
            instanceCount=len(self.children),  # uint32_t
            geometryCount=0,  # uint32_t
            pGeometries=None,  # const VkGeometryNV*
        )


class Geometry(Sinode):
    def __init__(self, setupDict, blas, initialMesh):
        Sinode.__init__(self, setupDict, blas)
        buffSetupDict = {}
        buffSetupDict["vertex"] = [[0, 1, 0], [1, 1, 1], [1, 1, 0]]
        buffSetupDict["index"] = [[0, 1, 2]]
        buffSetupDict["aabb"] = [[0, 1, 2]]
        self.vertexBuffer = Buffer(
            self.lookUp("device"), buffSetupDict["vertex"].flatten()
        )
        self.indexBuffer = Buffer(
            self.lookUp("device"), buffSetupDict["index"].flatten()
        )
        self.aabb = Buffer(self.lookUp("device"), buffSetupDict["aabb"].flatten())

        # ccw rotation
        theta = 0
        self.vkTransformMatrix = VkTransformMatrixKHR(
            # float    matrix[3][4];
            [cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1]
        )

        self.geometryTriangles = VkGeometryTrianglesNV(
            sType=VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV,
            pNext=None,
            vertexData=self.buffer.vkBuffer,
            vertexOffset=0,
            vertexCount=len(buffSetupDict["vertex"].flatten()),
            vertexStride=12,
            vertexFormat=VK_FORMAT_R32G32B32_SFLOAT,
            indexData=self.indexBuffer.vkBuffer,
            indexOffset=0,
            indexCount=len(buffSetupDict["index"].flatten()),
            indexType=VK_INDEX_TYPE_UINT32,
            transformData=self.vkTransformMatrix,
            transformOffset=0,
        )

        self.aabbs = VkGeometryAABBNV(
            sType=VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV,
            pNext=None,
            aabbData=self.aabb.vkBuffer,
            numAABBs=1,
            stride=4,
            offset=0,
        )

        self.geometryData = VkGeometryDataNV(
            triangles=self.geometryTriangles, aabbs=self.aabbs
        )

        # possible flags:

        # VK_GEOMETRY_OPAQUE_BIT_KHR = 0x00000001,
        # VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR = 0x00000002,
        # // Provided by VK_NV_ray_tracing
        # VK_GEOMETRY_OPAQUE_BIT_NV = VK_GEOMETRY_OPAQUE_BIT_KHR,
        # // Provided by VK_NV_ray_tracing
        # VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_NV

        # VK_GEOMETRY_OPAQUE_BIT_KHR indicates that this geometry does
        # not invoke the any-hit shaders even if present in a hit group.

        self.vkGeometry = VkGeometryNV(
            sType=VK_STRUCTURE_TYPE_GEOMETRY_NV,
            pNext=None,
            geometryType=VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            geometry=self.geometryData,
            flags=VK_GEOMETRY_OPAQUE_BIT_KHR,
        )


# If type is VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV then instanceCount must be 0
class BLASNV(AccelerationStructureNV):
    def __init__(self, setupDict, shader, initialMesh):
        AccelerationStructureNV.__init__(self, setupDict, shader)

        self.geometry = Geometry(initialMesh, self)

        # Provided by VK_NV_ray_tracing
        self.asInfo = VkAccelerationStructureInfoNV(
            sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV,
            pNext=None,  # const void*
            type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
            instanceCount=0,  # uint32_t
            geometryCount=1,  # uint32_t
            pGeometries=[self.geometry.vkGeometry],  # const VkGeometryNV*
        )


class AccelerationStructureKHR(AccelerationStructure):
    def __init__(self, setupDict, shader):
        AccelerationStructure.__init__(self, setupDict, shader)

        # Identify the above data as containing opaque triangles.
        asGeom = VkAccelerationStructureGeometryKHR(
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            geometryType=VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            flags=VK_GEOMETRY_OPAQUE_BIT_KHR,
            triangles=geometry.triangles,
        )

        # The entire array will be used to build the BLAS.
        offset = VkAccelerationStructureBuildRangeInfoKHR(
            firstVertex=0, primitiveCount=53324234, primitiveOffset=0, transformOffset=0
        )

        # Provided by VK_NV_ray_tracing
        pCreateInfo = VkAccelerationStructureCreateInfoKHR(
            sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV,  # VkStructureType
            pNext=None,  # const void*
            compactedSize=642000,  # VkDeviceSize
        )

        # Provided by VK_NV_ray_tracing
        self.vkAccelerationStructure = vkCreateAccelerationStructureNV(
            device=self.vkDevice, pCreateInfo=self.asCreateInfo, pAllocator=None
        )


# If type is VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_ then instanceCount must be 0
class BLAS(AccelerationStructure):
    def __init__(self, setupDict, shader, initialMesh):
        AccelerationStructure.__init__(self, setupDict, shader)

        self.geometry = Geometry(initialMesh, self)

        # Provided by VK__ray_tracing
        self.asInfo = VkAccelerationStructureInfo(
            sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_,
            pNext=None,  # const void*
            type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
            instanceCount=0,  # uint32_t
            geometryCount=1,  # uint32_t
            pGeometries=[self.geometry.vkGeometry],  # const VkGeometry*
        )


# If type is VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_ then geometryCount must be 0
class TLAS(AccelerationStructure):
    def __init__(self, setupDict, shader):
        AccelerationStructure.__init__(self, setupDict, shader)

        for blasName, blasDict in setupDict["blas"].items():
            newBlas = BLAS(blasDict, shader)
            self.children += [newBlas]

        # Provided by VK__ray_tracing
        self.asInfo = VkAccelerationStructureInfo(
            sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_,
            pNext=None,  # const void*
            type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
            instanceCount=len(self.children),  # uint32_t
            geometryCount=0,  # uint32_t
            pGeometries=None,  # const VkGeometry*
        )
