#!/bin/bash

# Upload the optimized training script
echo "Uploading optimized train_gpt_single.py..."
scp train_gpt_single.py root@$1:/workspace/yxanul_v1/

# SSH and run the training
echo "Starting training with optimized parameters..."
ssh root@$1 << 'EOF'
cd /workspace/yxanul_v1
echo "Configuration:"
echo "- Sequence length: 64K tokens"  
echo "- Gradient accumulation: 4 steps"
echo "- Effective batch size: 256K tokens"
echo "- Available VRAM: 32GB"
echo ""
echo "Starting training..."
python train_gpt_single.py
EOF