# snpe-docker
Docker that builds snpe 1.47 for onnx 

First download and place the snpe 1.47 zip in the root of this project, then, to build and run the docker with your snpe environment run the following:

```
docker build -t snpe .
docker run -it snpe
$ snpe-onnx-to-dlc -h
usage: snpe-onnx-to-dlc [-h] [--input_network INPUT_NETWORK] [-o OUTPUT_PATH]
                        [--copyright_file COPYRIGHT_FILE]
                        [--model_version MODEL_VERSION]
                        [--disable_batchnorm_folding]
                        [--input_type INPUT_NAME INPUT_TYPE]
                        [--input_encoding INPUT_NAME INPUT_ENCODING]
                        [--validation_target RUNTIME_TARGET PROCESSOR_TARGET]
                        [--strict] [--debug [DEBUG]]
                        [--dry_run [DRY_RUN]]

Script to convert onnxmodel into a DLC file.

optional arguments:
  -h, --help            show this help message and exit

required arguments:
  --input_network INPUT_NETWORK, -i INPUT_NETWORK
                        Path to the source framework model.

...
```
