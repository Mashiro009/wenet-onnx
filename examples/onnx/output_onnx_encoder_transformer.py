# 该文件是用来尝试输出forward chunk的
# 完成transformer forward chunk的encoder的onnx的代码转化
# 有小问题在于required_cache_size暂时还不能设置
# decode chunk by chunk is ok
# 应该是因为使用了该tensor进行了Bool判断 导致tracing失败
import argparse
from posixpath import join
from numpy import mod
import torch
import os
import yaml
import math
# Mixing tracing and scripting
import sys
import onnx
import onnxruntime
import numpy as np
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()



def output_encoder_transformer_onnx(encoder_model,encoder_model_path):
    # following is output onnx_encoder model code
    inputs = [torch.randn(1,60*(i+1),80) for i in range(5)]
    dummy_input1 = inputs[0]
    offset = torch.tensor(1,dtype=torch.int64)
    required_cache_size = torch.tensor(-1,dtype=torch.int64)
    # subsampling_cache=None
    # elayers_output_cache=None
    # conformer_cnn_cache=None
    subsampling_cache=torch.rand(1,1,256)
    elayers_output_cache=torch.rand(12,1,1,256)
    conformer_cnn_cache=torch.rand(12,1,256,15)
    torch.onnx.export(encoder_model,
                    (dummy_input1,
        offset,
        required_cache_size,
        subsampling_cache,
        elayers_output_cache,
        conformer_cnn_cache),
                    encoder_model_path,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input','offset','required_cache_size', 'i1', 'i2', 'i3'],
                    output_names=['output', 'o1', 'o2', 'o3'],
                    dynamic_axes={'input': [1], 'i1':[1], 'i2':[2],
                                    'output': [1], 'o1':[1], 'o2':[2]},
                    verbose=True
                    )
    # above is output onnx_encoder model code


def check_encoder_onnx_and_pytorch(encoder_model,encoder_model_path):
    # following is test torch encoder's function forward_chunk_onnx code
    inputs = [torch.randn(1,60*(i+1),80) for i in range(5)]
    offset = torch.tensor(1,dtype=torch.int64)
    required_cache_size = torch.tensor(-1,dtype=torch.int64)
    subsampling_cache=torch.rand(1,1,256)
    elayers_output_cache=torch.rand(12,1,1,256)
    conformer_cnn_cache=torch.rand(12,1,256,15)
    chunk_onnx_outputs = []
    for i in range(5):
        dummy_input1 = inputs[i]
        out1,subsampling_cache,elayers_output_cache,conformer_cnn_cache = encoder_model(dummy_input1,
            offset,
            required_cache_size,
            subsampling_cache,
            elayers_output_cache,
            conformer_cnn_cache)
        chunk_onnx_outputs.append(out1)
        offset += out1.size(1)
    # above is test torch encoder's function forward_chunk_onnx code

    # following is test torch encoder's function forward_chunk code
    # need to set_onnx_mode(False)
    encoder_model.forward = encoder_model.forward_chunk
    encoder_model.set_onnx_mode(False)
    offset = torch.tensor(0,dtype=torch.int64)
    required_cache_size = torch.tensor(-1,dtype=torch.int64)
    subsampling_cache=None
    elayers_output_cache=None
    conformer_cnn_cache=None
    torch_outputs = []

    for i in range(5):
        dummy_input1 = inputs[i]
        out2,subsampling_cache,elayers_output_cache,conformer_cnn_cache = encoder_model(dummy_input1,
            offset,
            required_cache_size,
            subsampling_cache,
            elayers_output_cache,
            conformer_cnn_cache)
        torch_outputs.append(out2)
        offset += out2.size(1)

    # above is test torch encoder's function forward_chunk code



    onnx_model = onnx.load(encoder_model_path)
    onnx.checker.check_model(onnx_model)
    # Print a human readable representation of the graph
    onnx.helper.printable_graph(onnx_model.graph)
    print("encoder onnx_model check pass!")


    # 下面的代码是用来测试生产的onnx是否与本身的pytorch结果保持一致

    ort_session = onnxruntime.InferenceSession(encoder_model_path)
    # print(len(ort_session.get_inputs()))
    print(encoder_model_path + " onnx model has " + str(len(ort_session.get_inputs())) + " args")

    # prepare data
    offset = torch.tensor(1,dtype=torch.int64)
    required_cache_size = torch.tensor(-1,dtype=torch.int64)
    subsampling_cache=torch.rand(1,1,256)
    elayers_output_cache=torch.rand(12,1,1,256)
    conformer_cnn_cache=torch.rand(12,1,256,15)
    offset = to_numpy(offset)
    subsampling_cache = to_numpy(subsampling_cache)
    elayers_output_cache=to_numpy(elayers_output_cache)
    for i in range(5):
        # compute ONNX Runtime output prediction
        dummy_input1 = inputs[i]
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input1),
                    ort_session.get_inputs()[1].name: offset,
                    # ort_session.get_inputs()[2].name: to_numpy(required_cache_size),
                    ort_session.get_inputs()[2].name: subsampling_cache,
                    ort_session.get_inputs()[3].name: elayers_output_cache,
                    #   ort_session.get_inputs()[5].name: to_numpy(conformer_cnn_cache)
                    }
        ort_outs = ort_session.run(None, ort_inputs)
        offset += ort_outs[0].shape[1]
        subsampling_cache = ort_outs[1]
        elayers_output_cache = ort_outs[2]
        np.testing.assert_allclose(to_numpy(torch_outputs[i]), ort_outs[0], rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(to_numpy(chunk_onnx_outputs[i]), ort_outs[0], rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(to_numpy(chunk_onnx_outputs[i]), to_numpy(torch_outputs[i]), rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")







def main():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_dir', required=True, help='checkpoint model')
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
    encoder_model = model.encoder
    encoder_model.forward = encoder_model.forward_chunk_onnx
    
    encoder_model_path=os.path.join(output_dir,'encoder_chunk.onnx')
    output_encoder_transformer_onnx(encoder_model,encoder_model_path)
    check_onnx = True
    if check_onnx:
        check_encoder_onnx_and_pytorch(encoder_model,encoder_model_path)


if __name__ == '__main__':
    main()




