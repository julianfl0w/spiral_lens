import json
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sinode"))
)
import sinode.sinode as sinode

import vulkan as vk
import numpy as np
import time

here = os.path.dirname(os.path.abspath(__file__))


def glsltype2python(glsltype):
    if glsltype == "float":
        return np.float32
    if glsltype == "float32_t":
        return np.float32
    elif glsltype == "float64_t":
        return np.float64
    elif glsltype == "double":
        return np.float64
    elif glsltype == "int":
        return np.int32
    elif glsltype == "uint":
        return np.uint32
    elif "vec" in glsltype:
        return np.float32
    else:
        print("type")
        print(glsltype)
        die


def glsltype2bytesize(glsltype):
    if glsltype == "float":
        return 4
    elif glsltype == "float32_t":
        return 4
    elif glsltype == "float64_t":
        return 8
    elif glsltype == "double":
        return 8
    elif glsltype == "int":
        return 4
    elif glsltype == "uint":
        return 4
    elif glsltype == "vec2":
        # return 12
        return 4
    elif glsltype == "vec3":
        # return 12
        return 4
    elif glsltype == "vec4":
        # return 16
        return 4
    else:
        raise Exception("Unrecognized type: " + glsltype)


class Buffer(sinode.Sinode):
    currLocation = 0

    def __str__(self):
        outstr = ""
        if hasattr(self, "name"):
            outstr += self.name
        else:
            outstr += str(type(self))

        return outstr

    def __init__(self, **kwargs):
        sinode.Sinode.__init__(self, parent=self.device, **kwargs)

        # set defaults
        self.proc_kwargs(
            overwrite=False,
            DEBUG=False,
            format=vk.VK_FORMAT_R64_SFLOAT,
            readFromCPU=True,
            usage=vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            memProperties=0
            | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            qualifier="",
            memtype="float",
            rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
            stride=4,
            compress=True,
            released=False,
            shape=[]
        )

        self.proc_kwargs(**kwargs)

        self.device = self.fromAbove("device")
        self.device.buffers += [self]
        self.vkDevice = self.device.vkDevice
        self.itemSizeBytes = glsltype2bytesize(self.memtype)
        self.vectorMultiplier = 1
        if self.memtype.startswith("vec"):
            self.vectorMultiplier = int(self.memtype[3])

        self.pythonType = glsltype2python(self.memtype)
        self.getSkipval()

        # for vec3 etc, the size is already bakd in
        self.itemCount = int(np.prod(self.shape))
        if self.memtype == "vec4":
            self.itemCount *= 4
        if self.memtype == "vec3":
            self.itemCount *= 3
        self.sizeBytes = int(self.itemCount * self.itemSizeBytes * self.skipval)

        self.debug("creating buffer " + self.name)

        # We will now create a buffer with these options
        self.bufferCreateInfo = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=self.sizeBytes,  # buffer size in bytes.
            usage=self.usage,  # buffer is used as a storage buffer.
            sharingMode=self.sharingMode,  # buffer is exclusive to a single queue family at a time.
        )
        self.debug(self.vkDevice)
        self.debug(self.bufferCreateInfo)
        self.vkBuffer = vk.vkCreateBuffer(self.vkDevice, self.bufferCreateInfo, None)

        # But the buffer doesn't allocate memory for itself, so we must do that manually.

        # First, we find the memory requirements for the buffer.
        memoryRequirements = vk.vkGetBufferMemoryRequirements(
            self.vkDevice, self.vkBuffer
        )

        # There are several types of memory that can be allocated, and we must choose a memory type that:
        # 1) Satisfies the memory requirements(memoryRequirements.memoryTypeBits).
        # 2) Satifies our own usage requirements. We want to be able to read the buffer memory from the GPU to the CPU
        #    with vkMapMemory, so we set vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.
        # Also, by setting vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory written by the device(GPU) will be easily
        # visible to the host(CPU), without having to call any extra flushing commands. So mainly for convenience, we set
        # this flag.
        index = self.device.findMemoryType(
            memoryRequirements.memoryTypeBits, self.memProperties
        )

        if index < 0:
            raise Exception("Requested memory type not available on this device")

        # Now use obtained memory requirements info to allocate the memory for the buffer.
        self.allocateInfo = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=memoryRequirements.size,  # specify required memory.
            memoryTypeIndex=index,
        )

        # allocate memory on device.
        self.debug("allocating")
        self.vkDeviceMemory = vk.vkAllocateMemory(
            self.vkDevice, self.allocateInfo, None
        )

        self.debug("done allocating")

        self.debug("mapping")
        # Map the buffer memory, so that we can read from it on the CPU.
        self.pmap = vk.vkMapMemory(
            device=self.vkDevice,
            memory=self.vkDeviceMemory,
            offset=0,
            size=self.sizeBytes,
            flags=0,
        )
        self.debug("done mapping")

        # these debug prints take forever
        # self.debug(len(self.pmap[:]))
        # self.debug(len(np.zeros((self.itemCount * self.skipval), dtype=self.pythonType)))

        # sometimes you may want to unmap from CPU
        if not self.readFromCPU:
            vk.vkUnmapMemory(self.vkDevice, self.vkDeviceMemory)
            self.pmap = None

        self.debug("binding to device")
        # Now associate that allocated memory with the buffer. With that, the buffer is backed by actual memory.
        vk.vkBindBufferMemory(
            device=self.vkDevice,
            buffer=self.vkBuffer,
            memory=self.vkDeviceMemory,
            memoryOffset=0,
        )
        self.debug("done binding to device")

        self.vkMappedMemoryRange = vk.VkMappedMemoryRange(
            sType=vk.VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            pNext=None,
            memory=self.vkDeviceMemory,
            offset=0,
            size=int(self.sizeBytes),
        )

        # initialize to zero
        self.zeroInitialize()
        self.flush()
        self.debug("done initializing")

        # NEEDED FOR RAYTRACING, FAILS BEFORE VULKAN 1.3
        # self.bufferDeviceAddressInfo = VkBufferDeviceAddressInfo(
        #    sType=vk.VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        #    pNext=None,
        #    buffer=self.vkBuffer,
        # )

        # Maintain an address pointer for feed operations
        self.addrPtr = 0

    def flush(self):
        return vk.vkFlushMappedMemoryRanges(
            device=self.device.vkDevice,
            memoryRangeCount=1,
            pMemoryRanges=[self.vkMappedMemoryRange],
        )

    # in some cases, memory access from the shader must be in increments of 16 bytes
    # so if we have a 4-byte float, we need to skip every 4th memory element
    def getSkipval(self):
        if (
            self.compress
            and self.usage == vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            and (
                self.memtype == "float64_t"
                or self.memtype == "float"
                or self.memtype == "uint"
                or self.memtype == "int"
                or self.memtype == "vec2"
                or self.memtype == "vec3"
                or self.memtype == "vec4"
                or self.memtype == "double"
            )
        ):
            self.skipval = 1
        elif (
            not self.usage & vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
            and not self.usage & vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT
        ) and self.itemSizeBytes <= 8:
            self.skipval = int(16 / self.itemSizeBytes)
        # vec3s (12 bytes) does not divide evenly into 16 :(
        # elif self.itemSize == 12:
        #    self.skipval = 4.0/3
        else:
            self.skipval = int(1)

    def debugSizeParams(self):
        self.debug("itemCount " + str(self.itemCount))
        self.debug("itemSize " + str(self.itemSizeBytes))
        self.debug("skipval " + str(self.skipval))
        self.debug("sizeBytes " + str(self.sizeBytes))

    def zeroInitialize(self, flush=True):
        if self.skipval == 1:
            self.pmap[:] = np.zeros((self.itemCount), dtype=self.pythonType)
        else:
            self.debug(len(self.pmap))
            self.debug(self.pythonType)
            # self.pmap[:] = np.zeros((self.itemCount), dtype=self.pythonType)
            self.set(np.zeros((self.itemCount), dtype=self.pythonType))
        if flush:
            self.flush()

    def oneInitialize(self):
        self.set(np.ones((self.itemCount), dtype=self.pythonType))

    def get(self, asComplex=False, flat=False, order="C"):
        # glsl to python
        flatArray = np.frombuffer(self.pmap, self.pythonType)
        # because GLSL only allows 16-byte access,
        # we need to skip a few values in the memory
        if asComplex:
            rcvdArray = list(flatArray.astype(float))
            rcvdArrayReal = rcvdArray[::4]
            rcvdArrayImag = 1j * flatArray[1::4].astype(complex)
            rcvdArrayComplex = rcvdArrayReal + rcvdArrayImag
            # finally, reshape according to the expected dims
            rcvdArray = np.array(rcvdArrayComplex).reshape(self.shape)
        elif self.memtype == "vec2":
            rcvdArrayList = list(flatArray.astype(float))
            rcvdArray = np.zeros(self.shape + [2])
            rcvdArray = np.append(
                np.expand_dims(rcvdArrayList[::4], 1),
                np.expand_dims(rcvdArrayList[1::4], 1),
                axis=1,
            )

        else:
            if self.compress:
                if flat:
                    rcvdArray = flatArray
                else:
                    rcvdArray = flatArray.reshape(self.shape, order=order)

            else:
                indices = np.arange(0, len(flatArray), self.skipval).astype(int)
                rcvdArray = np.array(flatArray[indices]).reshape(self.shape)
        return rcvdArray

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
        self.debug("releasing buffer " + self.name)
        if not self.released:
            self.debug("destroying buffer " + self.name)
            vk.vkFreeMemory(self.vkDevice, self.vkDeviceMemory, None)
            vk.vkDestroyBuffer(self.vkDevice, self.vkBuffer, None)
            self.released = True

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
                + self.memtype
                + " "
                + self.name
                + ";\n"
            )

    def getComputeDeclaration(self):
        if self.usage == vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT:
            b = "uniform "
            std = "std140"
        else:
            b = "buffer "
            if self.compress:
                std = "std430"
            else:
                std = "std140"

        if self.shape == []:
            retstr = (
                "layout("
                + std
                + ", set = "
                + str(self.descriptorSetBinding)
                + ", binding = "
                + str(self.binding)
                # + ", "
                # + "xfb_stride = " + str(self.stride)
                + ") "
                + b
                + self.name
                + "_buf\n{\n   "
                + self.qualifier
                + " "
                + self.memtype
                + " "
                + self.name
                + ";\n};\n"
            )

        else:
            retstr = (
                "layout("
                + std
                + ", set = "
                + str(self.descriptorSetBinding)
                + ", binding = "
                + str(self.binding)
                # + ", "
                # + "xfb_stride = " + str(self.stride)
                + ") "
                + b
                + self.name
                + "_buf\n{\n   "
                + self.qualifier
                + " "
                + self.memtype
                + " "
                + self.name
                + "["
                + str(int(self.sizeBytes / self.itemSizeBytes))
                + "];\n};\n"
            )

        return retstr

    def write(self, data):
        startByte = self.addrPtr
        endByte = self.addrPtr + len(data) * self.itemSizeBytes
        self.addrPtr = endByte

        self.pmap[startByte:endByte] = data
        return startByte

    def setByIndexVec(self, index, data):
        # self.debug(self.name + " setting " + str(index) + " to " + str(data))
        startByte = index * self.itemSizeBytes * self.skipval
        self.pmap[startByte : startByte + 4] = np.real(data).astype(np.float32)
        self.pmap[startByte + 4 : startByte + 8] = np.imag(data).astype(np.float32)

        # self.debug("setting " + str(index) + " to " + str(np.real(data).astype(np.float32)))
        # self.debug("setting " + str(index) + ".i to " + str(np.imag(data).astype(np.float32)))

    def setByIndex(self, index, data):
        # self.debug(self.name + " setting " + str(index) + " to " + str(data))
        startByte = index * self.itemSizeBytes * self.skipval
        endByte = index * self.itemSizeBytes * self.skipval + self.itemSizeBytes
        self.pmap[startByte:endByte] = np.array(data, dtype=self.pythonType)

    def setByIndexStart(self, startIndex, data):
        # if self.skipval != 1:
        #    raise ("You can only do this with new-format storage buffers!")
        # self.debug(self.name + " setting " + str(index) + " to " + str(data))
        startByte = startIndex * self.itemSizeBytes * self.skipval
        endByte = (
            startIndex * self.itemSizeBytes * self.skipval
            + self.itemSizeBytes * len(data)
        )
        self.pmap[startByte:endByte] = np.array(data, dtype=self.pythonType)

    def getByIndex(self, index):
        # self.debug(self.name + " setting " + str(index) + " to " + str(data))
        startByte = index * self.itemSizeBytes * self.skipval
        endByte = index * self.itemSizeBytes * self.skipval + self.itemSizeBytes
        return np.frombuffer(self.pmap[startByte:endByte], dtype=self.pythonType)

    def set(self, data, flush=True):
        # self.pmap[:] = data.astype(self.pythonType)

        if len(self.pmap[:]) != np.prod(data.shape) * self.itemSizeBytes:
            self.debug("WRONG SIZE")
            self.debug("pmap (bytes): " + str(len(self.pmap[:])))
            self.debug("item size (bytes): " + str(self.itemSizeBytes))
            self.debug(self.sizeBytes)
            self.debug("data (bytes): " + str(np.prod(data.shape) * self.itemSizeBytes))
            raise Exception("Wrong Size")

        if self.skipval == 1:
            # self.pmap[:] = data.astype(self.pythonType).flatten()
            self.pmap[:] = data.astype(self.pythonType).flatten()
        else:
            indices = np.arange(0, len(data), 1.0 / self.skipval).astype(int)
            data = data[indices]
            # print(self.pythonType)
            self.pmap[:] = data.astype(self.pythonType).flatten()

        if flush:
            self.flush()

    def fill(self, value):
        # self.pmap[: data.size * data.itemSize] = data
        a = np.array([value])
        # self.pmap[:a] = data[:int(a/(data.itemSize))]
        self.pmap[:] = np.tile(a, int(len(self.pmap[:]) / a.itemSize))

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


class StorageBuffer(Buffer):
    def __init__(self, **kwargs):
        self.parent = kwargs["device"]

        sinode.Sinode.__init__(self, **kwargs)

        # set defaults first
        self.setDefaults(
            overwrite=False,
            DEBUG=False,
            qualifier="",
            memProperties=0
            | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            memtype="float",
            rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
            compress=True,
            location=0,
            format=vk.VK_FORMAT_R64_SFLOAT,
            readFromCPU=True,
            usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            stride=16,
        )

        Buffer.__init__(self, **kwargs)

        # if "descriptorSet" not in kwargs.keys():
        #    self.descriptorSet = self.fromAbove("descriptorPool").descSetGlobal


class DebugBuffer(StorageBuffer):
    def __init__(self, **kwargs):
        sinode.Sinode.__init__(self, **kwargs)

        self.proc_kwargs(
            DEBUG=True,
            location=0,
            format=vk.VK_FORMAT_R64_SFLOAT,
            readFromCPU=True,
            memProperties=0
            | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            qualifier="",
            memtype="vec3",
            rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
            stride=12,
            compress=True,
        )
        StorageBuffer.__init__(self, **kwargs)


class VertexBuffer(Buffer):
    def __init__(
        self,
        device,
        name,
        shape,
        location,
        DEBUG=False,
        qualifier="",
        memProperties=0
        | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        memtype="float",
        rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
        stride=12,
        compress=False,
    ):
        # self.location = Buffer.currLocation
        # Buffer.currLocation+=1

        Buffer.__init__(
            self,
            DEBUG=False,
            device=device,
            name=name,
            location=location,
            shape=shape,
            format=vk.VK_FORMAT_R32G32B32_SFLOAT,
            readFromCPU=True,
            usage=vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            memProperties=memProperties,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
            qualifier=qualifier,
            memtype=memtype,
            rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
            stride=stride,
            compress=compress,
        )
        self.binding = location
        # the following are only needed for vertex buffers
        # vk.VK_VERTEX_INPUT_RATE_VERTEX: Move to the next data entry after each vertex
        # vk.VK_VERTEX_INPUT_RATE_INSTANCE: Move to the next data entry after each instance

        # we will standardize its bindings with a attribute description
        self.attributeDescription = vk.VkVertexInputAttributeDescription(
            binding=self.binding, location=self.location, format=self.format, offset=0
        )
        # ^^ Consider vk.VK_FORMAT_R32G32B32A32_SFLOAT  ?? ^^
        self.bindingDescription = vk.VkVertexInputBindingDescription(
            binding=self.binding, stride=stride, inputRate=rate  # 4 bytes/element
        )


class IndexBuffer(Buffer):
    def __init__(
        self,
        device,
        shape,
        DEBUG=False,
        qualifier="",
        memProperties=0
        | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        # | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
        stride=4,
    ):
        Buffer.__init__(
            self,
            DEBUG=False,
            device=device,
            name="index",
            location=0,
            shape=shape,
            format=vk.VK_FORMAT_R32_UINT,
            readFromCPU=True,
            usage=vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            memProperties=memProperties,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            stageFlags=vk.VK_SHADER_STAGE_ALL_GRAPHICS,
            qualifier=qualifier,
            memtype="uint",
            rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
            stride=4,
            compress=True,
        )


class FragmentBuffer(Buffer):
    def __init__(
        self,
        device,
        name,
        shape,
        DEBUG=False,
        qualifier="",
        memProperties=0
        | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        memtype="float",
        rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
        stride=12,
        compress=False,
    ):
        Buffer.__init__(
            self,
            DEBUG=False,
            device=device,
            name=name,
            location=0,
            shape=shape,
            format=vk.VK_FORMAT_R64_SFLOAT,
            readFromCPU=True,
            usage=vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            memProperties=memProperties,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            qualifier=qualifier,
            memtype=memtype,
            rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
            stride=stride,
            compress=compress,
        )


class UniformBuffer(Buffer):
    def __init__(
        self,
        device,
        name,
        shape,
        DEBUG=False,
        qualifier="",
        memProperties=0
        | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        memtype="float",
        rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
        stride=12,
    ):
        Buffer.__init__(
            self,
            DEBUG=False,
            device=device,
            name=name,
            location=0,
            shape=shape,
            format=vk.VK_FORMAT_R64_SFLOAT,
            readFromCPU=True,
            usage=vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            memProperties=memProperties,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            qualifier=qualifier,
            memtype=memtype,
            rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
            stride=12,
            compress=False,
        )


class AccelerationStructure(Buffer):
    def __init__(self, setupDict, shader):
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
        # 											compactSizes.data(), sizeof(VkDeviceSize), vk.VK_QUERY_RESULT_WAIT_BIT);

        # just playing. we will guess that b***h

        # Provided by vk.VK_NV_ray_tracing
        self.asCreateInfo = vk.VkAccelerationStructureCreateInfoNV(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV,
            pNext=None,
            compactedSize=642000,  # VkDeviceSize
        )

        # Provided by vk.VK_NV_ray_tracing
        self.vkAccelerationStructure = vk.vkCreateAccelerationStructureNV(
            device=self.vkDevice, pCreateInfo=self.asCreateInfo, pAllocator=None
        )


# If type is vk.VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV then geometryCount must be 0
class TLASNV(AccelerationStructureNV):
    def __init__(self, setupDict, shader):
        AccelerationStructureNV.__init__(self, setupDict, shader)

        for blasName, blasDict in setupDict["blas"].items():
            newBlas = BLASNV(blasDict, shader)
            self.children += [newBlas]

        # Provided by vk.VK_NV_ray_tracing
        self.asInfo = vk.VkAccelerationStructureInfoNV(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV,
            pNext=None,  # const void*
            type=vk.VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            flags=vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
            instanceCount=len(self.children),  # uint32_t
            geometryCount=0,  # uint32_t
            pGeometries=None,  # const VkGeometryNV*
        )


class Geometry(sinode.Sinode):
    def __init__(self, setupDict, blas, initialMesh):
        sinode.Sinode.__init__(self, setupDict, blas)
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
        self.vkTransformMatrix = vk.VkTransformMatrixKHR(
            # float    matrix[3][4];
            [np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta), 0, 0, 0, 1]
        )

        self.geometryTriangles = vk.VkGeometryTrianglesNV(
            sType=vk.VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV,
            pNext=None,
            vertexData=self.buffer.vkBuffer,
            vertexOffset=0,
            vertexCount=len(buffSetupDict["vertex"].flatten()),
            vertexStride=12,
            vertexFormat=vk.VK_FORMAT_R32G32B32_SFLOAT,
            indexData=self.indexBuffer.vkBuffer,
            indexOffset=0,
            indexCount=len(buffSetupDict["index"].flatten()),
            indexType=vk.VK_INDEX_TYPE_UINT32,
            transformData=self.vkTransformMatrix,
            transformOffset=0,
        )

        self.aabbs = vk.VkGeometryAABBNV(
            sType=vk.VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV,
            pNext=None,
            aabbData=self.aabb.vkBuffer,
            numAABBs=1,
            stride=4,
            offset=0,
        )

        self.geometryData = vk.VkGeometryDataNV(
            triangles=self.geometryTriangles, aabbs=self.aabbs
        )

        # possible flags:

        # vk.VK_GEOMETRY_OPAQUE_BIT_KHR = 0x00000001,
        # vk.VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR = 0x00000002,
        # // Provided by vk.VK_NV_ray_tracing
        # vk.VK_GEOMETRY_OPAQUE_BIT_NV = vk.VK_GEOMETRY_OPAQUE_BIT_KHR,
        # // Provided by vk.VK_NV_ray_tracing
        # vk.VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_NV

        # vk.VK_GEOMETRY_OPAQUE_BIT_KHR indicates that this geometry does
        # not invoke the any-hit shaders even if present in a hit group.

        self.vkGeometry = vk.VkGeometryNV(
            sType=vk.VK_STRUCTURE_TYPE_GEOMETRY_NV,
            pNext=None,
            geometryType=vk.VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            geometry=self.geometryData,
            flags=vk.VK_GEOMETRY_OPAQUE_BIT_KHR,
        )


# If type is vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV then instanceCount must be 0
class BLASNV(AccelerationStructureNV):
    def __init__(self, setupDict, shader, initialMesh):
        AccelerationStructureNV.__init__(self, setupDict, shader)

        self.geometry = Geometry(initialMesh, self)

        # Provided by vk.VK_NV_ray_tracing
        self.asInfo = vk.VkAccelerationStructureInfoNV(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV,
            pNext=None,  # const void*
            type=vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            flags=vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
            instanceCount=0,  # uint32_t
            geometryCount=1,  # uint32_t
            pGeometries=[self.geometry.vkGeometry],  # const VkGeometryNV*
        )


class AccelerationStructureKHR(AccelerationStructure):
    def __init__(self, setupDict, shader):
        AccelerationStructure.__init__(self, setupDict, shader)

        # Identify the above data as containing opaque triangles.
        asGeom = vk.VkAccelerationStructureGeometryKHR(
            vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            geometryType=vk.VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            flags=vk.VK_GEOMETRY_OPAQUE_BIT_KHR,
            triangles=geometry.triangles,
        )

        # The entire array will be used to build the BLAS.
        offset = vk.VkAccelerationStructureBuildRangeInfoKHR(
            firstVertex=0, primitiveCount=53324234, primitiveOffset=0, transformOffset=0
        )

        # Provided by vk.VK_NV_ray_tracing
        pCreateInfo = vk.VkAccelerationStructureCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV,  # VkStructureType
            pNext=None,  # const void*
            compactedSize=642000,  # VkDeviceSize
        )

        # Provided by vk.VK_NV_ray_tracing
        self.vkAccelerationStructure = vk.vkCreateAccelerationStructureNV(
            device=self.vkDevice, pCreateInfo=self.asCreateInfo, pAllocator=None
        )


# If type is vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_ then instanceCount must be 0
class BLAS(AccelerationStructure):
    def __init__(self, setupDict, shader, initialMesh):
        AccelerationStructure.__init__(self, setupDict, shader)

        self.geometry = Geometry(initialMesh, self)

        # Provided by vk.VK__ray_tracing
        self.asInfo = vk.VkAccelerationStructureInfo(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_,
            pNext=None,  # const void*
            type=vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            flags=vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
            instanceCount=0,  # uint32_t
            geometryCount=1,  # uint32_t
            pGeometries=[self.geometry.vkGeometry],  # const VkGeometry*
        )


# If type is vk.VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_ then geometryCount must be 0
class TLAS(AccelerationStructure):
    def __init__(self, setupDict, shader):
        AccelerationStructure.__init__(self, setupDict, shader)

        for blasName, blasDict in setupDict["blas"].items():
            newBlas = BLAS(blasDict, shader)
            self.children += [newBlas]

        # Provided by vk.VK__ray_tracing
        self.asInfo = vk.VkAccelerationStructureInfo(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_,
            pNext=None,  # const void*
            type=vk.VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            flags=vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
            instanceCount=len(self.children),  # uint32_t
            geometryCount=0,  # uint32_t
            pGeometries=None,  # const VkGeometry*
        )
