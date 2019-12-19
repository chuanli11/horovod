# Lambda Notes

### Installation

```
wget https://download.pytorch.org/whl/cu100/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl
wget https://download.pytorch.org/whl/cu100/torchvision-0.4.0-cp36-cp36m-manylinux1_x86_64.whl
pip3 install torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl
pip3 install torchvision-0.4.0-cp36-cp36m-manylinux1_x86_64.whl
rm torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl
rm torchvision-0.4.0-cp36-cp36m-manylinux1_x86_64.whl

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
HOROVOD_NCCL_HOME=/usr/local HOROVOD_GPU_ALLREDUCE=NCCL pip3 install --no-cache-dir horovod


# https://github.com/horovod/horovod/issues/1590
git clone --branch v0.18.2 https://github.com/horovod/horovod.git
```

### Pytorch distributed training with a single GPU node (MNIST)

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

mpirun -np 4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 pytorch_mnist.py \
    --batch-size 1024 |& tee 'pytorch_mnist_single_node_log.txt'
```

### Pytorch distributed training with multiple GPU nodes (MNIST)

```
mpirun -np 8 -H 10.1.10.187:4,10.1.10.43:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -x NCCL_SOCKET_IFNAME=^lo,docker0 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_exclude lo,docker0 \
    python3 pytorch_mnist.py \
    --batch-size 1024 |& tee 'pytorch_mnist_multi_node_log.txt'
```