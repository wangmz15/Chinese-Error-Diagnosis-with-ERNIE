import paddle, sys
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from model.transformer_encoder import *

# a = layers.fill_constant(shape=[2,3],dtype='float32',value=[[0.15849551, 0.45865775, 0.8563702 ],[0.12070083, 0.28766365, 0.18776911]])





# a1 = layers.fill_constant(shape=[2,1], dtype='float32', value=0.4667189)
# a2 = layers.fill_constant(shape=[2,1], dtype='float32', value=0.2058744)
# a3 = layers.fill_constant(shape=[2,1], dtype='float32', value=0.1548324)
# a4 = layers.fill_constant(shape=[2,1], dtype='float32', value=0.1725742)
a1 = layers.fill_constant(shape=[2,1], dtype='float32', value=0.18209726)
a2 = layers.fill_constant(shape=[2,1], dtype='float32', value=0.22834154)
# a3 = layers.fill_constant(shape=[2,1], dtype='float32', value=0.1)
a4 = layers.fill_constant(shape=[2,1], dtype='float32', value=0.20303878)
a = layers.concat([a1,a2, a4],axis=1,name='data')
indices = layers.argmax(a, axis=1)
# out,indices = layers.argsort(a, axis=-1)

cpu = fluid.core.CPUPlace()
exe = fluid.Executor(cpu)
# exe.run(fluid.default_startup_program(), fetch_list=[a.name])
exe.run(fluid.default_startup_program()) #

outs = exe.run(fetch_list=[a.name, indices.name])
for out in outs:
    print(out)