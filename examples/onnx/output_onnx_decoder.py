# 该文件是用来尝试输出decoder的
import argparse
from numpy import mod
import torch
import os
import yaml
# Mixing tracing and scripting
import sys
import onnx
import onnxruntime
import numpy as np
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def output_decoder_onnx(decoder_model,decoder_model_path):
    inputs = [torch.randn(16,60*(i+1),256) for i in range(5)]
    encoder_masks = [torch.ones(16,1,60*(i+1),dtype=torch.bool) for i in range(5)]
    hyps_pad = (abs(torch.randn(16, 30))*1000).ceil().long()
    hyps_lens = torch.arange(0,
                                32,
                                step=2,
                                dtype=torch.long)
    hyps_lens[0] = 1
    # following is output onnx_decoder model code
    dummy_input1= inputs[0]
    encoder_mask = encoder_masks[0]
    torch.onnx.export(decoder_model,
                    (dummy_input1, encoder_mask, hyps_pad,hyps_lens),
                    decoder_model_path,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input','encoder_mask', 'hyps_pad','hyps_lens'],
                    output_names=['output','o1','olens'],
                    dynamic_axes={'input': {0:'batch_size',1:'subsample_len'},
                                    'encoder_mask':{0:'batch_size',2:'subsample_len'},
                                    'hyps_pad':{0:'batch_size',1:'hyp_max_len'},
                                    'hyps_lens':{0:'batch_size'},
                                    'output': {0:'batch_size',1:'hyp_max_len'},
                                    'olens':{0:'batch_size',1:'hyp_max_len'},
                                    # 'show_x':{0:'batch_size',1:'hyp_max_len'}
                                    },
                    verbose=True
                    )

def check_decoder_onnx_and_pytorch(decoder_model,decoder_model_path):
    # following is test torch decoder's function forward code
    inputs = [torch.randn(16,60*(i+1),256) for i in range(5)]
    encoder_masks = [torch.ones(16,1,60*(i+1),dtype=torch.bool) for i in range(5)]
    hyps_pad = (abs(torch.randn(16, 30))*1000).ceil().long()
    hyps_lens = torch.arange(0,
                                32,
                                step=2,
                                dtype=torch.long)
    # inputs = [torch.randn(20,60*(i+1),256) for i in range(5)]
    # encoder_masks = [torch.ones(20,1,60*(i+1),dtype=torch.bool) for i in range(5)]
    # hyps_pad = (abs(torch.randn(20, 38))*1000).ceil().long()
    # hyps_lens = torch.arange(0,
    #                             40,
    #                             step=2,
    #                             dtype=torch.long)

    hyps_lens[0] = 1
    torch_outputs = []
    torch_shows = []
    for i in range(5):
        # compute ONNX Runtime output prediction
        dummy_input1 = inputs[i]
        encoder_mask = encoder_masks[i]
        decoder_out,_ , _ = decoder_model(dummy_input1, encoder_mask, hyps_pad,hyps_lens)
        torch_outputs.append(decoder_out)
        # torch_shows.append(try_x)
    # above is test torch decoder's function forward code


    # following is another test torch decoder's function forward code for check result whether close
    torch_outputs_another = []
    for i in range(5):
        # compute ONNX Runtime output prediction
        dummy_input1 = inputs[i]
        encoder_mask = encoder_masks[i]
        decoder_out1,_ , _ = decoder_model(dummy_input1, encoder_mask, hyps_pad,hyps_lens)
        torch_outputs_another.append(decoder_out1)
        np.testing.assert_allclose(to_numpy(torch_outputs[i]), to_numpy(torch_outputs_another[i]), rtol=1e-03, atol=1e-05)
    # above is another test torch decoder's function forward code for check result whether close


    
    onnx_model = onnx.load(decoder_model_path)
    onnx.checker.check_model(onnx_model)
    # Print a human readable representation of the graph
    onnx.helper.printable_graph(onnx_model.graph)
    print("decoder onnx_model check pass!")


    # 下面的代码是用来测试生产的onnx是否与本身的pytorch结果保持一致
    import onnxruntime

    ort_session = onnxruntime.InferenceSession(decoder_model_path)
    # print(len(ort_session.get_inputs()))
    print(decoder_model_path + " onnx model has " + str(len(ort_session.get_inputs())) + " args")

    for i in range(5):
        # compute ONNX Runtime output prediction
        dummy_input1 = inputs[i]
        encoder_mask = encoder_masks[i]
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input1),
                    ort_session.get_inputs()[1].name: to_numpy(encoder_mask),
                    ort_session.get_inputs()[2].name: to_numpy(hyps_pad),
                    ort_session.get_inputs()[3].name: to_numpy(hyps_lens),
                    }
        ort_outs = ort_session.run(None, ort_inputs)
        np.testing.assert_allclose(to_numpy(torch_outputs[i]), ort_outs[0], rtol=1e-03, atol=1e-05,err_msg='{0}'.format(i))
        # np.testing.assert_allclose(to_numpy(torch_shows[i]), ort_outs[3], rtol=1e-03, atol=1e-05,err_msg='{0}'.format(i))


    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    

def main():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_dir', required=True, help='output dir')
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
    decoder_model = model.decoder
    decoder_model.eval()
    
    decoder_model_path=os.path.join(output_dir,'decoder.onnx')
    output_decoder_onnx(decoder_model,decoder_model_path)
    check_onnx = True
    if check_onnx:
        check_decoder_onnx_and_pytorch(decoder_model,decoder_model_path)


if __name__ == '__main__':
    main()
