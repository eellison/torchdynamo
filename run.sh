# Setup the output directory
rm -rf /scratch/eellison/work/torchdynamo/benchmarks/bench_logs/torchbench_train/
mkdir /scratch/eellison/work/torchdynamo/benchmarks/bench_logs/torchbench_train/

# Commands for torchbench for device=cuda, dtype=float16 for training
python benchmarks/torchbench.py --float16 -dcuda --no-skip --output=/scratch/eellison/work/torchdynamo/benchmarks/bench_logs/torchbench_train//ts_nvfuser_torchbench_float16_training_cuda.csv --training --nvfuser --speedup-dynamo-ts --use-eval-mode --isolate --log-operator-inputs
python benchmarks/torchbench.py --float16 -dcuda --no-skip --output=/scratch/eellison/work/torchdynamo/benchmarks/bench_logs/torchbench_train//aot_nvfuser_torchbench_float16_training_cuda.csv --training --nvfuser --accuracy-aot-ts-mincut --use-eval-mode --isolate --log-operator-inputs
python benchmarks/torchbench.py --float16 -dcuda --no-skip --output=/scratch/eellison/work/torchdynamo/benchmarks/bench_logs/torchbench_train//inductor_cudagraphs_torchbench_float16_training_cuda.csv --training --inductor --use-eval-mode --isolate --log-operator-inputs

