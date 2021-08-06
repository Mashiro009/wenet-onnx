# WeNet

## ONNX

[当前仓库代码](https://github.com/Mashiro009/wenet-onnx)在于魔改原始[WeNet](https://github.com/wenet-e2e/wenet)源码，使其可以进行onnx导出

## Quick start

配置好examples/onnx/run_onnx.sh里面的config,checkpoint,output_dir

默认使用transformer
``` sh
cd examples/onnx/
./run_onnx.sh
```

导出的模型可以使用[wenet online onnx decoder](https://github.com/Mashiro009/wenet-online-decoder-onnx)代码进行测试


## 现在可以完成的工作

* 现只进行了required_cache_size=-1的测试（即所有的cache均要被使用）
* 测试了required_cache_size==0的情况，基本正常
* required_cache_size>0的情况无法运行
* 导出transformer的encoder,decoder,ctc 并测试
* 导出unified_transformer的encoder,decoder,ctc 并测试
* 导出unified_conformer的encoder,decoder,ctc 未测试

## 修改基本思路

参考原始[wenet](https://github.com/wenet-e2e/wenet)源码

参考作业帮给出部分的[方案](https://zhuanlan.zhihu.com/p/389441591)

所有的对Tensor取size的操作均不被onnx支持，所以要把其包在@torch.jit.script指示的函数里


## 对文件的修改记录

- wenet/utils/mask.py

  因onnx不支持tri算子,替换了subsequent_mask函数.

- wenet/transformer/subsampling.py
  
  对外不再使用position_encoding函数，改为position_encoding_onnx函数。原始forward_chunk调用传来的是x.size，现直接把x(Tensor)传给embedding层计算position_encoding.

- wenet/transformer/encoder.py
  1. line 123 添加了set_onnx_mode用于设置不同的代码流程
  2. line 325 重写了一份forward_chunk_onnx函数
    
     因为按作业帮的推送解释,onnx不能处理List[Tensor],所以对所有的cache(encoder_layer_cache、cnn_cache)均加了一维（从三维到四维）,第一维是原始的List的index.

     因为按作业帮的推送解释,onnx不能处理NoneType,所以在第一次处理时不能默认三个cache为None,所以要导入一份冗余的cache(本代码全部设为zeros),同时初始offset就被设为1（而不是0）.

     所有的cache在被处理时均被设为在时间轴上多了一（冗余）维,并携带冗余维进入encoder_layer（在每个layer层中自己切掉第一维）

     subsampling_cache (batch_size=1, time, feature_dim=256)

     elayers_output_cache (encoder_layer_num=12, batch_size=1, time, feature_dim=256)

     conformer_cnn_cache (encoder_layer_num=12, batch_size=1, feature_dim=256, time=15) 这个15跟convolution层的self.lorder有关,self.lorder = kernel_size - 1
  
- wenet/transformer/encoder_layer.py
  1. line 18 添加了slice_helper2用于辅助切片
  2. line 56 添加了self.set_onnx_mode(False)默认mode==False
  3. line 94 添加了切片处理,与wenet/transformer/encoder.py联动,防止cache多出来的冗余那一时间维参与计算
  4. line 116 使用添加的slice_helper2进行切片
  5. line 153 生产假的cnn_cache,并且恢复被第三项切掉的冗余时间维
  6. line 162 set_onnx_mode函数的定义
  7. ConformerEncoderLayer line 226,253,291 原理同transformer一致
  8. line 341 与transformer不同的是,该处的new_cnn_cache是有意义的,也要恢复冗余时间维
  9. line 156 更换了fake_cnn_cache的赋值,让transformer encoder onnx也能有cnn_cache的输入

- wenet/transformer/embedding.py
  1. line 48,59 PositionalEncoding forward的offset默认值调整为-1，是因为decoder的forward调用embedding时传不进来offset,使用的是默认值,而offset的类型不能再使用int（onnx的要求），所以使用了判断是否为-1（int）调整为0（Tensor）的"巧计"
  2. line 65 使用了slice_helper,还是为了帮助切片
  3. line 92 添加了position_encoding_onnx,代码逻辑与position_encoding完全相同,与wenet/transformer/subsampling.py的修改联动,修改也只是为了能够切片.
  4. line 142 使用了slice_helper,还是为了帮助切片

- wenet/transformer/decoder.py
  1. line 115 添加了新的make_pad_mask函数,是因为int(lengths.max().item())又把Tensor转成int了,使用lengths.max()作为替代
  2. line 204 添加了set_onnx_mode函数的定义
  3. 应该还不支持BiTransformerDecoder

- wenet/transformer/decoder_layer.py
  1. 添加了一些没有意义的调试代码（已注释）,是为了测试为什么decoder有误差
  
- wenet/transformer/convolution.py
  1. 添加了一些注释
  2. line 119 更换了new_cache的赋值,让不使用因果卷积的conformer也能正常运行

- wenet/transformer/asr_model.py
  1. line 544 添加了set_onnx_mode函数

- runtime/core/decoder/torch_asr_decoder.cc runtime/core/decoder/torch_asr_decoder.h
  1. 早期理解作业帮方案时不够好,添加了一些代码,从没测试过,对导出onnx没有影响,建议以后重写


## 添加的文件

- examples/onnx/web.py 网页端显示onnx的具体样子 可视化
- examples/onnx/BAC009S0764W0121.wav 用于测试的音频 来自aishell
- examples/onnx/output_onnx_encoder_transformer.py 用于导出transformer的encoder
- examples/onnx/output_onnx_encoder_conformer.py 用于导出conformer的encoder
- examples/onnx/output_onnx_ctc.py 用于导出ctc
- examples/onnx/output_onnx_decoder.py 用于导出decoder
- examples/onnx/path.sh 配置环境
- examples/onnx/run_onnx.sh 总的流程

## 环境配置

环境配置请参考[wenet](https://github.com/wenet-e2e/wenet)

## 现存的可能的bug

* 导出的decoder.onnx与pytorch的decoder计算有一些些误差

``` sh
Traceback (most recent call last):
File "onnx_decoder.py", line 198, in <module>
np.testing.assert_allclose(to_numpy(torch_outputs[i]), ort_outs[0], rtol=1e-03, atol=1e-05,err_msg='{0}'.format(i))

AssertionError:
Not equal to tolerance rtol=0.001, atol=1e-05
Mismatched elements: 761 / 2839200 (0.0268%)
Max absolute difference: 0.00216645
Max relative difference: 7.25
```

* 总的run.sh流程里有一定的冗余,多次加载了pytorch的模型,后续进行修改