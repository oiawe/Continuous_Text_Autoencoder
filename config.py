from dataclasses import dataclass
            
@dataclass
class TrainConfig:
    dataset_path: str = 'datas/falcon_train'
    tokenizer_path: str = 'datas/tokenizer'

    batch_size: int = 48
    learning_rate: float = 1e-4
    max_steps: int =  80000
    model_save_path: str = './checkpoints/0'
    log_dir: str = './runs/0'
    log_interval: int = 32
    test_interval: int = 400
    save_interval: int = 1000
    warmup_steps: int = 100
    num_workers: int = 2

    max_grad_norm: float = 100.0
    chunk_size: int = 128