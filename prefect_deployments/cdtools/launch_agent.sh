conda activate cdtools_cuda
CUDA_AVAILABLE_DEVICES="$1" prefect agent start -p default-agent-pool -q cdtools --limit 1
