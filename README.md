run at  machine A

```python
python -m torch.distributed.launch --nproc_per_node=4 --nnode=2 --node_rank=0 --master_addr=A_ip_address master_port=29500 main.py
```

run at machine  B

```python
python -m torch.distributed.launch --nproc_per_node=4 --nnode=2 --node_rank=1 --master_addr=A_ip_address master_port=29500 main.py
```

