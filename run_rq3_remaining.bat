@echo off
call C:\CoopCalib\venv\Scripts\activate
cd C:\CoopCalib\models\socialvae

echo === eth_1500_v0 ===
python main.py --config config/eth.py --train C:\CoopCalib\data\splits\eth\split_1500\train --test data/eth/test --device cuda --lambda3 0.0 --ckpt C:\CoopCalib\experiments\results\socialvae\eth_1500_v0

echo === eth_full_v0 ===
python main.py --config config/eth.py --train C:\CoopCalib\data\splits\eth\split_full\train --test data/eth/test --device cuda --lambda3 0.0 --ckpt C:\CoopCalib\experiments\results\socialvae\eth_full_v0

echo === eth_500_v3 ===
python main.py --config config/eth.py --train C:\CoopCalib\data\splits\eth\split_500\train --test data/eth/test --device cuda --lambda3 0.1 --ckpt C:\CoopCalib\experiments\results\socialvae\eth_500_v3

echo === eth_1000_v3 ===
python main.py --config config/eth.py --train C:\CoopCalib\data\splits\eth\split_1000\train --test data/eth/test --device cuda --lambda3 0.1 --ckpt C:\CoopCalib\experiments\results\socialvae\eth_1000_v3

echo === eth_1500_v3 ===
python main.py --config config/eth.py --train C:\CoopCalib\data\splits\eth\split_1500\train --test data/eth/test --device cuda --lambda3 0.1 --ckpt C:\CoopCalib\experiments\results\socialvae\eth_1500_v3

echo === eth_full_v3 ===
python main.py --config config/eth.py --train C:\CoopCalib\data\splits\eth\split_full\train --test data/eth/test --device cuda --lambda3 0.1 --ckpt C:\CoopCalib\experiments\results\socialvae\eth_full_v3

echo === ALL RQ3 RUNS COMPLETE ===
pause