#!/bin/bash

. ./path.sh

# config=exp/transformer/train.yaml
# checkpoint=exp/transformer/81.pt
# output_dir=exp/transformer/onnx_model

model_dir=exp/transformer
config=$model_dir/train.yaml
# checkpoint=$model_dir/$(ls $model_dir | grep "pt") # 使用该命令请保证路径下只有一个pt文件
checkpoint=$model_dir/81.pt
output_dir=$model_dir/onnx_model

python ./output_onnx_encoder_transformer.py \
  --config $config \
  --checkpoint $checkpoint \
  --output_dir $output_dir

python ./output_onnx_ctc.py \
  --config $config \
  --checkpoint $checkpoint \
  --output_dir $output_dir

python ./output_onnx_decoder.py \
  --config $config \
  --checkpoint $checkpoint \
  --output_dir $output_dir


# config=exp/unified_conformer/train.yaml
# checkpoint=exp/unified_conformer/43.pt
# output_dir=exp/unified_conformer/onnx_model

# python ./output_onnx_encoder_conformer.py \
#   --config $config \
#   --checkpoint $checkpoint \
#   --output_dir $output_dir