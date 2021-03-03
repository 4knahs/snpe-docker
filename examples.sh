rsync -azP cbb2:EDSR/onnx_models/TPSR.onnx mount/model.onnx
# Check the README.md on how to generate benchmark_config.json
# All these examples are explained in the same document.
rm mount/model.dlc
./run.sh snpe-onnx-to-dlc -i /mnt/files/model.onnx
#./run.sh snpe-dlc-info -i /mnt/files/model.dlc
#./run.sh /mnt/files/run_benchmark.sh /mnt/files/benchmark_config.json


#### Quantized
rm mount/model_quantized.dlc
# 8 bit quantisation----------------
# ./run.sh snpe-dlc-quantize \
#     --input_dlc /mnt/files/model.dlc \
#     --input_list /mnt/files/input_data/input_files.txt \
#     --output_dlc /mnt/files/model_quantized.dlc \
#     --enable_hta \
#     --hta_partitions 2-28
# A16W8 bit quantisation----------------
./run.sh snpe-dlc-quantize \
    --input_dlc /mnt/files/model.dlc \
    --input_list /mnt/files/input_data/input_files.txt \
    --output_dlc /mnt/files/model_quantized.dlc \
    --enable_hta \
    --hta_partitions 2-28 \
    --act_bitwidth 16 --weights_bitwidth 8
#./run.sh snpe-dlc-info -i /mnt/files/model_quantized.dlc
./run.sh /mnt/files/run_benchmark.sh /mnt/files/benchmark_config_quantized.json