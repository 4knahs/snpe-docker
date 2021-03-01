# Check the README.md on how to generate benchmark_config.json
# All these examples are explained in the same document.
# ./run.sh snpe-onnx-to-dlc -i /mnt/files/model.onnx
# ./run.sh snpe-dlc-info -i /mnt/files/model.dlc
# ./run.sh /mnt/files/run_benchmark.sh /mnt/files/benchmark_config.json
# ./run.sh snpe-dlc-quantize --input_dlc /mnt/files/model.dlc --input_list /mnt/files/input_data/input_files.txt --output_dlc /mnt/files/model_quantized.dlc --enable_hta