import numpy as np, onnx, onnxruntime as ort, sys
from onnx import helper, TensorProto, numpy_helper
src = sys.argv[1]; dst = sys.argv[2]
s = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
x = np.random.rand(2, 78, 8, 8).astype(np.float32)
outs = s.run(None, {"data": x})
P = outs[1].shape[1]
print("policy size", P)
rng = np.random.default_rng(0)
def w(name, shape, scale=0.5):
    return numpy_helper.from_array((rng.standard_normal(shape) * scale).astype(np.float32), name)
inits = [w("Wa", (78, P), 2.0), w("Wb", (78, P), 2.0), w("Wv", (78, 1)), w("Wd", (78, 3)), w("Wm", (78, 1))]
inits.append(numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "axes"))
nodes = [
    helper.make_node("ReduceMean", ["data", "axes"], ["m"], keepdims=0),
    helper.make_node("MatMul", ["m", "Wa"], ["pi_a"]),
    helper.make_node("MatMul", ["m", "Wb"], ["pi_b"]),
    helper.make_node("MatMul", ["m", "Wv"], ["v0"]),
    helper.make_node("Tanh", ["v0"], ["value"]),
    helper.make_node("MatMul", ["m", "Wd"], ["d0"]),
    helper.make_node("Softmax", ["d0"], ["wdl_out"], axis=1),
    helper.make_node("MatMul", ["m", "Wm"], ["m0"]),
    helper.make_node("Relu", ["m0"], ["moves_left"]),
]
g = helper.make_graph(nodes, "dummy",
    [helper.make_tensor_value_info("data", TensorProto.FLOAT, ["batch", 78, 8, 8])],
    [helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch", 1]),
     helper.make_tensor_value_info("pi_a", TensorProto.FLOAT, ["batch", P]),
     helper.make_tensor_value_info("pi_b", TensorProto.FLOAT, ["batch", P]),
     helper.make_tensor_value_info("wdl_out", TensorProto.FLOAT, ["batch", 3]),
     helper.make_tensor_value_info("moves_left", TensorProto.FLOAT, ["batch", 1])], inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
m.ir_version = 8
onnx.checker.check_model(m)
onnx.save(m, dst)
