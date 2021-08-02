# 网页端显示onnx的具体样子 可视化
# 不建议可视化encoder和decoder的onnx,比较大,加载比较慢
import netron
modelPath = "./examples/onnx/loop.onnx"
netron.start(modelPath,address=12347)