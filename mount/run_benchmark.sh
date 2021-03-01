#!/usr/bin/env bash

BENCHMARK_CONFIG_FILE=$1 # /mnt/files/benchmark_config.json

adb devices

pushd $SNPE_ROOT/benchmarks
	ln -s ../lib/python/* . 2> /dev/null
	python snpe_bench.py -c $BENCHMARK_CONFIG_FILE -a
popd
