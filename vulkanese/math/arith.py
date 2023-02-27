import os
import sys
import time
import numpy as np

arith_home = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import vulkanese as ve
import vulkan as vk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sinode")))
import sinode.sinode as sinode

class ARITH(ve.shader.Shader):
    def __init__(
        self,
        **kwargs
    ):
        self.kwdefault = {

            "parent":None,
            "OPERATION":None,
            "FUNCTION1":None,
            "FUNCTION2":None,
            "npEquivalent":None,
            "DEBUG":False,
            "buffType":"float",
            "shader_basename":"shaders/arith",
            "memProperties":(
                vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            ),
            "useFence":True, 
        }
        sinode.Sinode.__init__(self, **kwargs)
        constantsDict = {}
        constantsDict["PROCTYPE"] = self.buffType
        if self.OPERATION is not None:
            constantsDict["OPERATION"] = self.OPERATION
        if self.FUNCTION1 is not None:
            constantsDict["FUNCTION1"] = self.FUNCTION1
        if self.FUNCTION2 is not None:
            constantsDict["FUNCTION2"] = self.FUNCTION2
        constantsDict["YLEN"] = np.prod(np.shape(self.Y))
        constantsDict["LG_WG_SIZE"] = 7  # corresponding to 128 threads, a good number
        constantsDict["THREADS_PER_WORKGROUP"] = 1 << constantsDict["LG_WG_SIZE"]

        # device selection and instantiation
        self.instance = self.device.instance
        self.constantsDict = constantsDict

        self.descriptorPool = ve.descriptor.DescriptorPool(device=self.device, parent = self)

        buffers = [
            ve.buffer.StorageBuffer(
                device=self.device,
                name="x",
                memtype=self.buffType,
                qualifier="readonly",
                dimensionVals=np.shape(self.X),
                memProperties=self.memProperties,
                parent=self,
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="y",
                memtype=self.buffType,
                qualifier="readonly",
                dimensionVals=np.shape(self.Y),
                memProperties=self.memProperties,
                parent=self,
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="sumOut",
                memtype=self.buffType,
                qualifier="writeonly",
                dimensionVals=np.shape(self.X),
                memProperties=self.memProperties,
                parent=self,
            ),
        ]

        self.descriptorPool.finalize()

        # Compute Stage: the only stage
        ve.shader.Shader.__init__(
            self,
            sourceFilename=os.path.join(
                arith_home, self.shader_basename + ".c"
            ),  # can be GLSL or SPIRV
            constantsDict=self.constantsDict,
            device=self.device,
            name=self.shader_basename,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            buffers=buffers,
            DEBUG=self.DEBUG,
            workgroupCount=[
                int(np.prod(np.shape(self.X)) / (constantsDict["THREADS_PER_WORKGROUP"])),
                1,
                1,
            ],
            useFence = self.useFence
        )

        self.gpuBuffers.x.set(self.X)
        self.gpuBuffers.y.set(self.Y)
        
    def baseline(self, X, Y):
        if self.OPERATION is not None:
            retval = self.npEquivalent(X,Y)
            return retval
        if self.FUNCTION1 is not None:
            return eval("np." + self.FUNCTION1  + "(X)")
        if self.FUNCTION2 is not None:
            return eval("np." + self.FUNCTION2  + "(X, Y)")
        else:
            die

    def test(self):

        self.run(blocking=True)
        result = self.gpuBuffers.sumOut.get()
        expectation = self.baseline(
            self.gpuBuffers.x.get(),
            self.gpuBuffers.y.get()
            )
        print("------------")
        print(expectation)
        print(self.gpuBuffers.sumOut.get())
        self.passed = np.allclose(result.astype(float), expectation.astype(float))
        if self.OPERATION is not None:
            print(self.OPERATION + ": " + str(self.passed))
        if self.FUNCTION1 is not None:
            print(self.FUNCTION1 + ": " + str(self.passed))
        if self.FUNCTION2 is not None:
            print(self.FUNCTION2 + ": " + str(self.passed))
        return self.passed


def test(device):
    print("Testing Arithmatic")
    signalLen = 2 ** 4
    X = np.random.random((signalLen))
    Y = np.random.random((signalLen))
    toTest = [
        ARITH(device = device, X=X, Y=Y, OPERATION="+"   , npEquivalent=np.add),
        #ARITH(device = device, X=X, Y=Y, OPERATION="-"   , npEquivalent=np.subtract),
        #ARITH(device = device, X=X, Y=Y, OPERATION="*"   , npEquivalent=np.multiply),
        #ARITH(device = device, X=X, Y=Y, OPERATION="/"   , npEquivalent=np.divide),
        #ARITH(device = device, X=X, Y=Y, FUNCTION1="sin" ),
        #ARITH(device = device, X=X, Y=Y, FUNCTION1="cos" ),
        #ARITH(device = device, X=X, Y=Y, FUNCTION1="tan" ),
        #ARITH(device = device, X=X, Y=Y, FUNCTION1="exp" ),
        ##ARITH(device = device, X=X, Y=Y, FUNCTION1="asin"),
        ##ARITH(device = device, X=X, Y=Y, FUNCTION1="acos"),
        ##ARITH(device = device, X=X, Y=Y, FUNCTION1="atan"),
        #ARITH(device = device, X=X, Y=Y, FUNCTION1="sqrt"),
        ##ARITH(device = device, X=X, Y=Y, FUNCTION2="pow" ),
        ##ARITH(device = device, X=X, Y=Y, FUNCTION2="mod" ),
        ##ARITH(device = device, X=X, Y=Y, FUNCTION2="atan"),
    ]
    for s in toTest:
        s.test()
        #s.release()


if __name__ == "__main__":

    # begin GPU test
    instance = ve.instance.Instance(verbose=False)
    device = instance.getDevice(0)
    
    test(device=device)
    instance.release()
