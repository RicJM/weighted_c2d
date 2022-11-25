from dataloaders import dataloader_cifar as dataloader

loader = dataloader.cifar_dataloader(
    'cifar100',
    r=0.2,
    noise_mode='sym',
    batch_size='32',
    num_workers=5,
    root_dir='cifar-100-python',
    log='',
    noise_file='cifar100_noise.json',
    stronger_aug=False,
)

