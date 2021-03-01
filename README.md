# snpe-docker
Docker container and helpers to builds snpe 1.47 for onnx model convertion and benchmarking.

Most examples assume that you are using vision models, but they can still help you with your task at hand.

## Build the docker container

First download and place the [snpe 1.47 zip](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk) in the root of this project, then, to build and run the docker with your snpe environment run the following:

```bash
$ git clone https://github.com/4knahs/snpe-docker.git
$ cd snpe-docker
$ # Download snpe 1.47 and place in this folder
$ docker build -t snpe .
```

To get a bash shell just do: 

```bash
docker run -v "$(pwd)"/mount:/mnt/files --privileged -v /dev/bus/usb:/dev/bus/usb -it snpe
```

Note that the above command:
* Mounts the `mount/` folder into `/mnt/files` in the container. This is to be able to access your models but also retrieve the results from snpe runs.
* Runs with priviledged mode and mounts `/dev/bus/usb`. This is to give the container visibility into adb devices connected via usb.

All the snpe binaries should be accessible (e.g., `snpe-onnx-to-dlc`, `snpe-dlc-info`, `snpe-dlc-quantize`, etc).

## ONNX to DLC

First, the onnx model (named `model.onnx`) should be placed in the `mount/` folder (i.e., `mount/model.onnx`).

Then, we included an helper script (`run.sh`) that basically runs the docker instance but also does a few more steps:
* Kill the local adb daemon - this is to ensure that adb devices are visible from within the container.
* Docker build - in case you missed the previous step.
* Docker run - this is the same as shown above, with the mount and priviledged properties.

To convert `mount/model.onnx` simply run:

```bash
./run.sh snpe-onnx-to-dlc -i /mnt/files/model.onnx
```

## Get DLC info

Assuming the previous step was run and that it generated a `model.dlc` file in the local `/mount/model.dlc`, just run:

```bash
./run.sh snpe-dlc-info -i /mnt/files/model.dlc
```

Note that the path `/mnt/files/model.dlc` is the path to the model as seen by the container (i.e., the mounted volume).

## Run benchmark on adb device

This process is a bit trickier as it requires multiple steps:
* Create or add input images.
* Create an `android_input_files.txt` file with references to your images.
* Create a benchmark config file.
* Run the benchmark.

To create a random input image we added a `create_raw_images.py` python script. Given a width (`-x`) and height (`-y`) this script generates a random image (BGR format) as follows:

```bash
mkdir mount/input_data
python create_raw_images.py -x 160 -y 90 -o mount/input_data/raw.img
```

If you are unsure about the format you should be using, check the previous dlc info step, the dlc convertion most likely converted your model to use BGR for vision tasks.

Now we need to reference the generated image from a `input_files.txt` file:

```bash
# Note: you can change the model name, but make sure to change everywhere
MODEL_NAME=AwesomeModel
# The following line works with multiple *.img files (change to whatever format you need)
ls mount/input_data/*.img | xargs -I[] echo "/data/local/tmp/snpebm/$MODEL_NAME/input_data/[] >> mount/input_data/android_input_files.txt"
```

Then create a benchmark config file (`mount/benchmark_config.json`) with the following content:

```bash
{
    "Name":"AwesomeModel",
    "HostRootPath": "AwesomeModel",
    "HostResultsDir":"AwesomeModel/results",
    "DevicePath":"/data/local/tmp/snpebm",
    "Devices":["b699ea80"],
    "HostName": "localhost",
    "Runs":1,

    "Model": {
        "Name": "AwesomeModel",
        "Dlc": "/mnt/files/model.dlc",
        "InputList": "/mnt/files/input_data/android_input_files.txt",
        "Data": [
            "/mnt/files/input_data"
        ]
    },

    "Runtimes":["GPU", "DSP", "AIP", "CPU"],
    "Measurements": ["timing"],
    "CpuFallback": true
 }
```

Obviously feel free to change any of the configurations to fit your needs.

Finally, to run the benchmark do the following:

```bash
./run.sh /mnt/files/run_benchmark.sh /mnt/files/benchmark_config.json
```

## Quantise DLC model

To quantise the previously generated model.dlc, we need two steps:
* Create an `input_files.txt`. This differs from the previous input file in the sense that paths are relative to the container instead of the adb device.
* Run quantisation.

To create the input_files.txt:

```bash
# The following line works with multiple *.img files (change to whatever format you need)
ls mount/input_data/*.img | xargs -I[] echo "/mnt/files/input_data/[] >> mount/input_data/input_files.txt"
```

To quantise the model do:

```bash
./run.sh snpe-dlc-quantize --input_dlc /mnt/files/model.dlc --input_list /mnt/files/input_data/input_files.txt --output_dlc /mnt/files/model_quantized.dlc --enable_hta
```

Your new model will be in `mount/model_quantized.dlc`. As expected you can now benchmark this model as well, just make sure that in the above steps you update `mount/benchmark_config.json` to:
* use `input_files.txt` instead of `android_input_files.txt`.
* use `model_quantized.dlc` instead of `model.dlc`