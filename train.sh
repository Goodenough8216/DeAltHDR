conda run -n py1.12 bash -c "CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train_dealthdr.py -opt options/DeAltHDR.yml 2>&1 | tee -a DeAltHDR_train.log"
