
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