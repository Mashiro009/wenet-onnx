# 该文件是用来尝试输出ctc的
import argparse
from examples.aishell.s0.tools.text2token import main
from numpy import mod
import torch
import os
import yaml
# Mixing tracing and scripting
import onnx
import onnxruntime
import numpy as np
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def output_ctc_onnx(ctc_model,ctc_model_path):
    # following is output onnx_ctc model code
    inputs = [torch.randn(1,60*(i+1),256) for i in range(5)]
    dummy_input1= inputs[0]
    torch.onnx.export(ctc_model,
                    (dummy_input1),
                    ctc_model_path,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': [1],
                                    'output': [1]},
                    verbose=True
                    )


def check_onnx_and_pytorch(ctc_model,ctc_model_path):
    
    inputs = [torch.randn(1,60*(i+1),256) for i in range(5)]
    # following 生成pytorch结果
    torch_outputs = []
    for i in range(5):
        # compute ONNX Runtime output prediction
        dummy_input1 = inputs[i]
        out1 = ctc_model(dummy_input1)
        torch_outputs.append(out1)
    # above 生成pytorch结果


    onnx_model = onnx.load(ctc_model_path)
    onnx.checker.check_model(onnx_model)
    # Print a human readable representation of the graph
    onnx.helper.printable_graph(onnx_model.graph)
    print("ctc onnx_model check pass!")

    # 下面的代码是用来测试生产的onnx是否与本身的pytorch结果保持一致
    import onnxruntime

    ort_session = onnxruntime.InferenceSession(ctc_model_path)
    print(ctc_model_path + " onnx model has " + str(len(ort_session.get_inputs())) + " args")

    for i in range(5):
        # compute ONNX Runtime output prediction
        dummy_input1 = inputs[i]
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input1),
                    }
        ort_outs = ort_session.run(None, ort_inputs)
        np.testing.assert_allclose(to_numpy(torch_outputs[i]), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

def main():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_dir', required=True, help='output_dir')
    args = parser.parse_args()
    output_dir=args.output_dir
    os.system("mkdir -p "+ output_dir)
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # following load model
    model = init_asr_model(configs)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    # above load model


    model.set_onnx_mode(True)
    ctc_model = model.ctc
    ctc_model.forward = ctc_model.log_softmax
    
    ctc_model_path=os.path.join(output_dir,'ctc.onnx')
    output_ctc_onnx(ctc_model,ctc_model_path)
    check_onnx = True
    if check_onnx:
        check_onnx_and_pytorch(ctc_model,ctc_model_path)


if __name__ == '__main__':
    main()