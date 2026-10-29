"""Run four CIFAR-10 seeds, followed by four CIFAR-100 seeds (Table 3)."""
from BLADE.run_tables_multiseed import main

if __name__ == '__main__':
    main('table3_cifar10_cifar100')
