Run with : python train_fp8_optimized.py --data_dir /workspace/yxanul_v1/mixed_6b/train.bin --vocab_size 32768 --n_layer 12 --n_head 12 --n_kv_heads 3 --n_embd 768 --seed 1337
==================================================
Starting CLEAN FP8 Training (Actual Fastest)
Model: 12L, 12H, 768D
Parameters: 99.5M
Status:

FP8: True
Gradient fusion: DISABLED (overhead)
Weight caching: NOT IMPLEMENTED (overhead > benefit)
Expected: 196k tokens/sec on RTX 5090
Batch size: 8
Gradient accumulation: 16
Effective batch size: 128
==================================================
iter 0: loss 10.5586, lr 0.00e+00, 183.3k tok/s, FP8: False
Step 0: val loss 10.5363
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 10: loss 9.2617, lr 4.00e-05, 176.9k tok/s, FP8: False
iter 20: loss 8.6562, lr 8.00e-05, 195.4k tok/s, FP8: False
iter 30: loss 7.9688, lr 1.20e-04, 195.3k tok/s, FP8: False
iter 40: loss 7.1973, lr 1.60e-04, 195.4k tok/s, FP8: False
iter 50: loss 6.7246, lr 2.00e-04, 195.1k tok/s, FP8: False
iter 60: loss 6.3633, lr 2.40e-04, 195.1k tok/s, FP8: False
iter 70: loss 6.0918, lr 2.80e-04, 194.8k tok/s, FP8: False
iter 80: loss 5.7656, lr 3.20e-04, 194.6k tok/s, FP8: False
iter 90: loss 5.7344, lr 3.60e-04, 194.5k tok/s, FP8: False
iter 100: loss 5.4668, lr 4.00e-04, 195.4k tok/s, FP8: True
iter 110: loss 5.4414, lr 4.40e-04, 224.4k tok/s, FP8: True
iter 120: loss 5.2461, lr 4.80e-04, 225.0k tok/s, FP8: True
iter 130: loss 5.3223, lr 5.20e-04, 224.4k tok/s, FP8: True
iter 140: loss 5.2383, lr 5.60e-04, 224.7k tok/s, FP8: True
iter 150: loss 5.1875, lr 6.00e-04, 224.3k tok/s, FP8: True
iter 160: loss 5.1074, lr 6.40e-04, 224.5k tok/s, FP8: True
iter 170: loss 5.0586, lr 6.80e-04, 224.2k tok/s, FP8: True
iter 180: loss 4.9492, lr 7.20e-04, 224.3k tok/s, FP8: True
iter 190: loss 4.8320, lr 7.60e-04, 224.0k tok/s, FP8: True
iter 200: loss 4.6543, lr 8.00e-04, 224.2k tok/s, FP8: True
Step 200: val loss 3.8291
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 210: loss 4.6094, lr 8.00e-04, 198.8k tok/s, FP8: True
iter 220: loss 4.7422, lr 8.00e-04, 223.6k tok/s, FP8: True
iter 230: loss 4.4180, lr 8.00e-04, 224.0k tok/s, FP8: True
iter 240: loss 4.4502, lr 8.00e-04, 223.9k tok/s, FP8: True
iter 250: loss 4.4248, lr 8.00e-04, 223.3k tok/s, FP8: True
iter 260: loss 4.2656, lr 8.00e-04, 223.6k tok/s, FP8: True
iter 270: loss 4.3018, lr 8.00e-04, 223.8k tok/s, FP8: True
iter 280: loss 4.1445, lr 8.00e-04, 223.1k tok/s, FP8: True
iter 290: loss 4.1074, lr 8.00e-04, 223.7k tok/s, FP8: True
iter 300: loss 4.1367, lr 8.00e-04, 223.5k tok/s, FP8: True
iter 310: loss 4.2910, lr 8.00e-04, 223.5k tok/s, FP8: True
iter 320: loss 4.2842, lr 8.00e-04, 223.1k tok/s, FP8: True
iter 330: loss 4.1416, lr 8.00e-04, 223.5k tok/s, FP8: True
iter 340: loss 4.0557, lr 8.00e-04, 222.9k tok/s, FP8: True
iter 350: loss 3.9941, lr 8.00e-04, 223.4k tok/s, FP8: True
iter 360: loss 3.9219, lr 8.00e-04, 223.2k tok/s, FP8: True
iter 370: loss 3.9746, lr 8.00e-04, 223.4k tok/s, FP8: True
iter 380: loss 3.9072, lr 8.00e-04, 223.1k tok/s, FP8: True
iter 390: loss 3.6758, lr 8.00e-04, 223.4k tok/s, FP8: True
iter 400: loss 3.6514, lr 8.00e-04, 223.1k tok/s, FP8: True
Step 400: val loss 2.6909
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 410: loss 3.8320, lr 8.00e-04, 199.3k tok/s, FP8: True
iter 420: loss 3.7227, lr 8.00e-04, 223.1k tok/s, FP8: True
iter 430: loss 3.6250, lr 8.00e-04, 223.1k tok/s, FP8: True
iter 440: loss 3.5889, lr 8.00e-04, 223.3k tok/s, FP8: True
iter 450: loss 3.6211, lr 8.00e-04, 223.3k tok/s, FP8: True
iter 460: loss 3.5977, lr 8.00e-04, 223.0k tok/s, FP8: True
iter 470: loss 3.4805, lr 8.00e-04, 223.1k tok/s, FP8: True
iter 480: loss 3.5137, lr 8.00e-04, 223.3k tok/s, FP8: True
iter 490: loss 3.5264, lr 8.00e-04, 222.8k tok/s, FP8: True
iter 500: loss 3.3682, lr 8.00e-04, 223.2k tok/s, FP8: True
Saving checkpoint to checkpoints_fp8_optimized/checkpoint_500_fp8_optimized.pt
iter 510: loss 3.4424, lr 8.00e-04, 218.3k tok/s, FP8: True
iter 520: loss 3.6396, lr 8.00e-04, 223.3k tok/s, FP8: True




TEST 2! 
python train_fp8_optimized.py --data_dir /workspace/yxanul_v1/mixed_6b/train.bin --vocab_size 32768 --n_layer 23 --n_head 12 --n_kv_heads 3 --n_embd 576 --seed 1337
==================================================
Starting CLEAN FP8 Training (Actual Fastest)
==================================================
Model: 23L, 12H, 576D
Parameters: 99.0M
Status:
  - FP8: True
  - Gradient fusion: DISABLED (overhead)
  - Weight caching: NOT IMPLEMENTED (overhead > benefit)
  - Expected: 196k tokens/sec on RTX 5090
Batch size: 8
Gradient accumulation: 16
Effective batch size: 128
==================================================
iter 0: loss 10.4961, lr 0.00e+00, 144.6k tok/s, FP8: False
Step 0: val loss 10.4637
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 10: loss 9.4922, lr 4.00e-05, 138.7k tok/s, FP8: False
iter 20: loss 9.0352, lr 8.00e-05, 153.0k tok/s, FP8: False
iter 30: loss 8.3359, lr 1.20e-04, 152.8k tok/s, FP8: False
iter 40: loss 7.6270, lr 1.60e-04, 152.7k tok/s, FP8: False
iter 50: loss 7.0215, lr 2.00e-04, 152.6k tok/s, FP8: False
iter 60: loss 6.6152, lr 2.40e-04, 152.5k tok/s, FP8: False
iter 70: loss 6.3301, lr 2.80e-04, 152.4k tok/s, FP8: False
iter 80: loss 6.0332, lr 3.20e-04, 152.6k tok/s, FP8: False
iter 90: loss 5.8555, lr 3.60e-04, 152.3k tok/s, FP8: False
iter 100: loss 5.8008, lr 4.00e-04, 152.5k tok/s, FP8: True
iter 110: loss 5.8301, lr 4.40e-04, 172.3k tok/s, FP8: True
iter 120: loss 5.5938, lr 4.80e-04, 172.1k tok/s, FP8: True
iter 130: loss 5.4785, lr 5.20e-04, 172.2k tok/s, FP8: True
iter 140: loss 5.2969, lr 5.60e-04, 171.9k tok/s, FP8: True
iter 150: loss 5.4297, lr 6.00e-04, 171.9k tok/s, FP8: True
iter 160: loss 5.1406, lr 6.40e-04, 172.0k tok/s, FP8: True
iter 170: loss 5.0781, lr 6.80e-04, 171.7k tok/s, FP8: True
iter 180: loss 5.0195, lr 7.20e-04, 171.6k tok/s, FP8: True
iter 190: loss 5.0293, lr 7.60e-04, 171.9k tok/s, FP8: True
iter 200: loss 4.9863, lr 8.00e-04, 171.6k tok/s, FP8: True
Step 200: val loss 4.0228
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 210: loss 4.7676, lr 8.00e-04, 153.7k tok/s, FP8: True
iter 220: loss 4.7500, lr 8.00e-04, 171.8k tok/s, FP8: True
iter 230: loss 4.7637, lr 8.00e-04, 171.5k tok/s, FP8: True
iter 240: loss 4.7227, lr 8.00e-04, 171.3k tok/s, FP8: True
iter 250: loss 4.7852, lr 8.00e-04, 171.7k tok/s, FP8: True
iter 260: loss 4.5762, lr 8.00e-04, 171.3k tok/s, FP8: True
iter 270: loss 4.3545, lr 8.00e-04, 171.6k tok/s, FP8: True
iter 280: loss 4.5566, lr 8.00e-04, 171.1k tok/s, FP8: True
iter 290: loss 4.3652, lr 8.00e-04, 171.5k tok/s, FP8: True
iter 300: loss 4.3066, lr 8.00e-04, 171.1k tok/s, FP8: True
iter 310: loss 4.3145, lr 8.00e-04, 171.4k tok/s, FP8: True
iter 320: loss 4.2402, lr 8.00e-04, 171.2k tok/s, FP8: True
iter 330: loss 4.1143, lr 8.00e-04, 170.9k tok/s, FP8: True
iter 340: loss 4.0254, lr 8.00e-04, 171.2k tok/s, FP8: True
iter 350: loss 4.0342, lr 8.00e-04, 170.9k tok/s, FP8: True
iter 360: loss 3.9473, lr 8.00e-04, 171.3k tok/s, FP8: True
iter 370: loss 3.9492, lr 8.00e-04, 170.8k tok/s, FP8: True
iter 380: loss 3.9473, lr 8.00e-04, 170.9k tok/s, FP8: True
iter 390: loss 4.0420, lr 8.00e-04, 171.2k tok/s, FP8: True
iter 400: loss 3.8760, lr 8.00e-04, 170.8k tok/s, FP8: True
Step 400: val loss 2.8353
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 410: loss 3.8525, lr 8.00e-04, 152.4k tok/s, FP8: True
iter 420: loss 3.6719, lr 8.00e-04, 171.4k tok/s, FP8: True
iter 430: loss 3.8936, lr 8.00e-04, 171.2k tok/s, FP8: True



TEST 3!!

python train_fp8_optimized.py --data_dir /workspace/yxanul_v1/mixed_6b/train.bin --vocab_size 32768 --n_layer 30 --n_head 8 --n_kv_heads 2 --n_embd 512 --seed 1337

==================================================
Starting CLEAN FP8 Training (Actual Fastest)
==================================================
Model: 30L, 8H, 512D
Parameters: 101.3M
Status:
  - FP8: True
  - Gradient fusion: DISABLED (overhead)
  - Weight caching: NOT IMPLEMENTED (overhead > benefit)
  - Expected: 196k tokens/sec on RTX 5090
Batch size: 8
Gradient accumulation: 16
Effective batch size: 128
==================================================

iter 0: loss 10.4961, lr 0.00e+00, 137.4k tok/s, FP8: False
Step 0: val loss 10.4425
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 10: loss 9.5625, lr 4.00e-05, 133.6k tok/s, FP8: False
iter 20: loss 9.1641, lr 8.00e-05, 148.1k tok/s, FP8: False
iter 30: loss 8.5273, lr 1.20e-04, 148.3k tok/s, FP8: False
iter 40: loss 7.8301, lr 1.60e-04, 148.5k tok/s, FP8: False
iter 50: loss 7.1328, lr 2.00e-04, 148.2k tok/s, FP8: False
iter 60: loss 6.6758, lr 2.40e-04, 148.2k tok/s, FP8: False
iter 70: loss 6.4941, lr 2.80e-04, 148.1k tok/s, FP8: False
iter 80: loss 6.2324, lr 3.20e-04, 148.0k tok/s, FP8: False
iter 90: loss 6.1895, lr 3.60e-04, 147.9k tok/s, FP8: False
iter 100: loss 5.8066, lr 4.00e-04, 148.5k tok/s, FP8: True
iter 110: loss 5.7832, lr 4.40e-04, 168.3k tok/s, FP8: Trueiter 120: loss 5.6387, lr 4.80e-04, 168.6k tok/s, FP8: True
iter 130: loss 5.5352, lr 5.20e-04, 168.3k tok/s, FP8: True
iter 140: loss 5.4297, lr 5.60e-04, 168.0k tok/s, FP8: True
iter 150: loss 5.1855, lr 6.00e-04, 168.7k tok/s, FP8: True
iter 160: loss 5.2871, lr 6.40e-04, 168.4k tok/s, FP8: True
iter 170: loss 5.0410, lr 6.80e-04, 168.7k tok/s, FP8: True
iter 180: loss 5.0488, lr 7.20e-04, 168.1k tok/s, FP8: True
iter 190: loss 4.9668, lr 7.60e-04, 168.1k tok/s, FP8: True
iter 200: loss 5.0059, lr 8.00e-04, 168.4k tok/s, FP8: True
Step 200: val loss 4.0375
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 210: loss 4.9492, lr 8.00e-04, 149.6k tok/s, FP8: True
iter 220: loss 4.7852, lr 8.00e-04, 168.1k tok/s, FP8: True
iter 230: loss 4.7637, lr 8.00e-04, 168.3k tok/s, FP8: True
iter 240: loss 4.6895, lr 8.00e-04, 168.6k tok/s, FP8: True
iter 250: loss 4.6592, lr 8.00e-04, 168.3k tok/s, FP8: True
iter 260: loss 4.5918, lr 8.00e-04, 168.4k tok/s, FP8: True
iter 270: loss 4.4922, lr 8.00e-04, 168.0k tok/s, FP8: True
iter 280: loss 4.3643, lr 8.00e-04, 168.1k tok/s, FP8: True
iter 290: loss 4.3564, lr 8.00e-04, 168.2k tok/s, FP8: True
iter 300: loss 4.3604, lr 8.00e-04, 167.6k tok/s, FP8: True
iter 310: loss 4.1914, lr 8.00e-04, 167.8k tok/s, FP8: True
iter 320: loss 4.2598, lr 8.00e-04, 168.2k tok/s, FP8: True
iter 330: loss 4.1309, lr 8.00e-04, 168.0k tok/s, FP8: True
iter 340: loss 4.0586, lr 8.00e-04, 168.2k tok/s, FP8: True
iter 350: loss 3.9629, lr 8.00e-04, 168.3k tok/s, FP8: True
iter 360: loss 4.0127, lr 8.00e-04, 167.8k tok/s, FP8: True
iter 370: loss 4.0303, lr 8.00e-04, 167.8k tok/s, FP8: True
iter 380: loss 3.9619, lr 8.00e-04, 168.2k tok/s, FP8: True
iter 390: loss 3.9023, lr 8.00e-04, 167.9k tok/s, FP8: True
iter 400: loss 3.9561, lr 8.00e-04, 167.8k tok/s, FP8: True
Step 400: val loss 2.8112
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 410: loss 3.7549, lr 8.00e-04, 150.2k tok/s, FP8: True
iter 420: loss 3.7412, lr 8.00e-04, 167.7k tok/s, FP8: True
iter 430: loss 3.8926, lr 8.00e-04, 167.8k tok/s, FP8: True
iter 440: loss 4.2168, lr 8.00e-04, 168.2k tok/s, FP8: True



TEST 4 !!
python train_fp8_optimized.py --data_dir /workspace/yxanul_v1/mixed_6b/train.bin --vocab_size 32768 --n_layer 38 --n_head 8 --n_kv_heads 2 --n_embd 512 --seed 1337
==================================================
Starting CLEAN FP8 Training (Actual Fastest)
==================================================
Model: 38L, 8H, 512D
Parameters: 123.9M
Status:
  - FP8: True
  - Gradient fusion: DISABLED (overhead)
  - Weight caching: NOT IMPLEMENTED (overhead > benefit)
  - Expected: 196k tokens/sec on RTX 5090
Batch size: 8
Gradient accumulation: 16
Effective batch size: 128
==================================================
iter 0: loss 10.5000, lr 0.00e+00, 114.7k tok/s, FP8: False
Step 0: val loss 10.4550
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 10: loss 9.5586, lr 4.00e-05, 109.3k tok/s, FP8: False
iter 20: loss 9.1172, lr 8.00e-05, 120.3k tok/s, FP8: False
iter 30: loss 8.6445, lr 1.20e-04, 120.2k tok/s, FP8: False
iter 40: loss 7.9258, lr 1.60e-04, 120.1k tok/s, FP8: False
iter 50: loss 7.2031, lr 2.00e-04, 119.9k tok/s, FP8: False
iter 60: loss 6.7480, lr 2.40e-04, 119.9k tok/s, FP8: False
iter 70: loss 6.5254, lr 2.80e-04, 119.9k tok/s, FP8: False
iter 80: loss 6.2227, lr 3.20e-04, 119.8k tok/s, FP8: False
iter 90: loss 6.1348, lr 3.60e-04, 119.9k tok/s, FP8: False
iter 100: loss 5.7969, lr 4.00e-04, 120.5k tok/s, FP8: True
iter 110: loss 5.7324, lr 4.40e-04, 136.8k tok/s, FP8: True
iter 120: loss 5.6328, lr 4.80e-04, 136.9k tok/s, FP8: True
iter 130: loss 5.5645, lr 5.20e-04, 137.3k tok/s, FP8: True
iter 140: loss 5.5117, lr 5.60e-04, 137.0k tok/s, FP8: True
iter 150: loss 5.3223, lr 6.00e-04, 137.3k tok/s, FP8: True
iter 160: loss 5.3789, lr 6.40e-04, 136.5k tok/s, FP8: True
iter 170: loss 5.1074, lr 6.80e-04, 136.9k tok/s, FP8: True
iter 180: loss 5.1035, lr 7.20e-04, 136.8k tok/s, FP8: True
iter 190: loss 5.0391, lr 7.60e-04, 137.3k tok/s, FP8: True
iter 200: loss 5.1094, lr 8.00e-04, 136.4k tok/s, FP8: True
Step 200: val loss 4.1347
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 210: loss 4.9805, lr 8.00e-04, 121.8k tok/s, FP8: True
iter 220: loss 4.8164, lr 8.00e-04, 136.6k tok/s, FP8: True
iter 230: loss 4.7949, lr 8.00e-04, 136.8k tok/s, FP8: True
iter 240: loss 4.7188, lr 8.00e-04, 136.4k tok/s, FP8: True
iter 250: loss 4.7217, lr 8.00e-04, 136.4k tok/s, FP8: True
iter 260: loss 4.6348, lr 8.00e-04, 136.4k tok/s, FP8: True
iter 270: loss 4.5195, lr 8.00e-04, 136.6k tok/s, FP8: True
iter 280: loss 4.4346, lr 8.00e-04, 136.6k tok/s, FP8: True
iter 290: loss 4.4072, lr 8.00e-04, 136.4k tok/s, FP8: True
iter 300: loss 4.4033, lr 8.00e-04, 136.7k tok/s, FP8: True
iter 310: loss 4.1953, lr 8.00e-04, 136.3k tok/s, FP8: True
iter 320: loss 4.3008, lr 8.00e-04, 136.6k tok/s, FP8: True
iter 330: loss 4.1494, lr 8.00e-04, 136.5k tok/s, FP8: True
iter 340: loss 4.0547, lr 8.00e-04, 136.4k tok/s, FP8: True
iter 350: loss 3.9385, lr 8.00e-04, 136.1k tok/s, FP8: True
iter 360: loss 3.9609, lr 8.00e-04, 136.3k tok/s, FP8: True
iter 370: loss 4.0059, lr 8.00e-04, 136.5k tok/s, FP8: True
iter 380: loss 3.9150, lr 8.00e-04, 136.5k tok/s, FP8: True
iter 390: loss 3.8330, lr 8.00e-04, 136.8k tok/s, FP8: True
iter 400: loss 3.9053, lr 8.00e-04, 136.4k tok/s, FP8: True
Step 400: val loss 2.7044
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 410: loss 3.6963, lr 8.00e-04, 121.2k tok/s, FP8: True
iter 420: loss 3.6641, lr 8.00e-04, 136.3k tok/s, FP8: True
iter 430: loss 3.6611, lr 8.00e-04, 136.6k tok/s, FP8: True




TEST 5 SOPHIA G ! ! 
python train_fp8_optimized.py --data_dir /workspace/yxanul_v1/mixed_6b/train.bin --vocab_size 32768 --n_layer 38 --n_head 8 --n_kv_heads 2 --n_embd 512 --opt sophia --sophia_lr 6e-4 --sophia_betas 0.965 0.99 --sophia_rho 0.05 --sophia_weight_decay 0.2 --sophia_k 10 --seed 1337



iter 0: loss 10.5000, lr 0.00e+00, 113.2k tok/s, FP8: False
Step 0: val loss 10.4550
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 10: loss 9.5547, lr 4.00e-05, 108.5k tok/s, FP8: False
iter 20: loss 9.1641, lr 8.00e-05, 119.3k tok/s, FP8: False
iter 30: loss 8.6289, lr 1.20e-04, 119.0k tok/s, FP8: False
iter 40: loss 8.0508, lr 1.60e-04, 118.9k tok/s, FP8: False
iter 50: loss 7.4863, lr 2.00e-04, 119.0k tok/s, FP8: False
iter 60: loss 7.2188, lr 2.40e-04, 119.0k tok/s, FP8: False
iter 70: loss 7.1074, lr 2.80e-04, 118.9k tok/s, FP8: False
iter 80: loss 7.0586, lr 3.20e-04, 118.8k tok/s, FP8: False
iter 90: loss 6.9688, lr 3.60e-04, 118.7k tok/s, FP8: False
iter 100: loss 6.7012, lr 4.00e-04, 119.7k tok/s, FP8: True
iter 110: loss 6.6699, lr 4.40e-04, 134.4k tok/s, FP8: True
iter 120: loss 6.5938, lr 4.80e-04, 135.6k tok/s, FP8: True
iter 130: loss 6.5117, lr 5.20e-04, 134.1k tok/s, FP8: True
iter 140: loss 6.4160, lr 5.60e-04, 135.2k tok/s, FP8: True
iter 150: loss 6.2461, lr 6.00e-04, 134.7k tok/s, FP8: True
iter 160: loss 6.2773, lr 6.40e-04, 134.7k tok/s, FP8: True
iter 170: loss 6.0703, lr 6.80e-04, 135.4k tok/s, FP8: True
iter 180: loss 6.0352, lr 7.20e-04, 135.3k tok/s, FP8: True
iter 190: loss 5.9570, lr 7.60e-04, 135.1k tok/s, FP8: True
iter 200: loss 5.9590, lr 8.00e-04, 135.6k tok/s, FP8: True
Step 200: val loss 5.1387
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 210: loss 5.8535, lr 8.00e-04, 120.4k tok/s, FP8: True
iter 220: loss 5.7480, lr 8.00e-04, 135.6k tok/s, FP8: True
iter 230: loss 5.7305, lr 8.00e-04, 134.8k tok/s, FP8: True
iter 240: loss 5.6641, lr 8.00e-04, 135.5k tok/s, FP8: True
iter 250: loss 5.6680, lr 8.00e-04, 134.7k tok/s, FP8: True
iter 260: loss 5.6113, lr 8.00e-04, 135.4k tok/s, FP8: True
iter 270: loss 5.5156, lr 8.00e-04, 135.7k tok/s, FP8: True
iter 280: loss 5.4043, lr 8.00e-04, 134.8k tok/s, FP8: True
iter 290: loss 5.4082, lr 8.00e-04, 134.9k tok/s, FP8: True
iter 300: loss 5.4258, lr 8.00e-04, 135.2k tok/s, FP8: True
iter 310: loss 5.2637, lr 8.00e-04, 135.3k tok/s, FP8: True
iter 320: loss 5.3164, lr 8.00e-04, 135.3k tok/s, FP8: True
iter 330: loss 5.2344, lr 8.00e-04, 134.9k tok/s, FP8: True
iter 340: loss 5.1719, lr 8.00e-04, 134.9k tok/s, FP8: True
iter 350: loss 5.1426, lr 8.00e-04, 135.4k tok/s, FP8: True
iter 360: loss 5.1875, lr 8.00e-04, 135.2k tok/s, FP8: True
iter 370: loss 5.1855, lr 8.00e-04, 134.8k tok/s, FP8: True
iter 380: loss 5.1504, lr 8.00e-04, 135.2k tok/s, FP8: True
iter 390: loss 5.1270, lr 8.00e-04, 135.1k tok/s, FP8: True
iter 400: loss 5.1855, lr 8.00e-04, 135.6k tok/s, FP8: True
Step 400: val loss 4.2888
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 410: loss 5.0547, lr 8.00e-04, 120.8k tok/s, FP8: True
iter 420: loss 5.0215, lr 8.00e-04, 135.5k tok/s, FP8: True
iter 430: loss 5.0449, lr 8.00e-04, 135.5k tok/s, FP8: True
iter 440: loss 5.0684, lr 8.00e-04, 135.2k tok/s, FP8: True
iter 450: loss 4.9863, lr 8.00e-04, 135.9k tok/s, FP8: True
iter 460: loss 4.9473, lr 8.00e-04, 134.8k tok/s, FP8: True
iter 470: loss 5.0215, lr 8.00e-04, 135.7k tok/s, FP8: True
iter 480: loss 4.9863, lr 8.00e-04, 135.4k tok/s, FP8: True
iter 490: loss 4.9824, lr 8.00e-04, 135.7k tok/s, FP8: True
iter 500: loss 5.0117, lr 8.00e-04, 135.7k tok/s, FP8: True
Saving checkpoint to checkpoints_fp8_optimized/checkpoint_500_fp8_optimized.pt

Here’s the math and concrete values for your current setup.

Effective tokens/iter: batch_size × grad_accum × seq_len

12 × 22 × 2048 = 540,672 tokens/iter
For ~6.0B tokens total

max_iters: ceil(6,000,000,000 / 540,672) ≈ 11,100
lr_decay_iters: set equal to max_iters → 11,100
With warmup_iters=2000, the schedule is:
Plateau: 0.6 × 11,100 = 6,660 steps (at base LR)
Decay: 11,100 − 2,000 − 6,660 = 2,440 steps (cosine to min_lr)
For full train split (~6,184,775,876 tokens)

max_iters: ceil(6,184,775,876 / 540,672) ≈ 11,440
lr_decay_iters: 11,440
Plateau: 0.6 × 11,440 = 6,864; Decay: 11,440 − 2,000 − 6,864 = 2,576
Quick formula for any future change

tokens_per_iter = B × accum × T
max_iters = ceil(total_tokens / tokens_per_iter)
lr_decay_iters = max_iters (to finish decay near the end)



Vanishing risk:
grad/first_block_rms < 1e-7 for many steps, ratio_last_first > 100.
Loss not improving while global grad_norm is small and clip_rate ~0.
Exploding risk:
clip_rate > 0.2 sustained, frequent loss spikes, max_block_rms outliers orders of magnitude above median.
update/weight ratios >> 1e-2 for many blocks (depends on LR and scale).
Rules of thumb

Vanishing: first_block_rms very small (e.g., <1e-7) while last_block_rms is much larger; ratio_last_first >> 1 for many steps.
Exploding: frequent clipping (clip_rate > 0.2 sustained), spikes in max_block_rms, or large upd/max_block_ratio.

lip rate

Tracks fraction of steps where grad_norm > grad_clip.
Logged as train/clip_rate.
Per-layer gradient RMS (sampled)

Computes RMS grad per parameter, aggregates to per-block (transformer.h.N.*).
Logs every 100 steps:
grad/first_block_rms, grad/last_block_rms, grad/ratio_last_first
grad/min_block_rms, grad/median_block_rms, grad/max_block_rms
Update/Weight ratio per block

Right after optimizer.step(), computes RMS(update)/RMS(weight) per parameter and aggregates to per-block.
Logs at every logging step:
upd/first_block_ratio, upd/last_block_ratio, upd/ratio_last_first
upd/min_block_ratio, upd/median_block_ratio, upd/max_block_ratio
Stores previous parameter snapshot per tensor in p._prev_data.

 python train_fp8_optimized.py --data_dir /workspace/yxanul_v1/mixed_6b/train.bin --vocab_size 32768 --n_layer 38 --n_head 8 --n_kv_heads 2 --n_embd 512 --opt sophia --sophia_lr 6e-4 --sophia_betas 0.965 0.99 --sophia_rho 0.05 --sophia_weight_decay 0.2 --sophia_k 10 --seed 1337 --batch_size 12 


ATTN LAYERS : python train_fp8_optimized.py --data_dir /workspace/yxanul_v1/mixed_6b/train.bin --vocab_size 32768 --n_layer 38 --n_head 8 --n_kv_heads 2 --n_embd 512 --opt sophia --sophia_lr 6e-4 --sophia_betas 0.965 0.99 --sophia_rho 0.05 --sophia_weight_decay 0.2 --sophia_k 10 --seed 1337 --batch_size 12 --attn_windowed --attn_window 256 --attn_global_layers 6,12,18,24,30,37 --attn_dilated_range 13:29 --attn_dilation 2 --attn_local_chunk 128




TEST 6 !!
python train_fp8_optimized.py --data_dir /workspace/yxanul_v1/mixed_6b/train.bin --vocab_size 32768 --n_layer 38 --n_head 8 --n_kv_heads 2 --n_embd 512 --opt sophia --sophia_lr 6e-4 --sophia_betas 0.965 0.99 --sophia_rho 0.05 --sophia_weight_decay 0.2 --sophia_k 10 --seed 1337 --batch_size 12 --attn_windowed --attn_window 256 --attn_global_layers 6,12,18,24,30,37 --attn_dilated_range 13:29 --attn_dilation 1 --attn_local_chunk 128

Optimizer: SophiaG
  SophiaG lr=0.0006, betas=(0.965, 0.99), rho=0.05, wd=0.2, k=10
Batch size: 12
Gradient accumulation: 22
Effective batch size: 264
==================================================

iter 0: loss 10.4972, lr 0.00e+00, 136.5k tok/s, FP8: False
Step 0: val loss 10.4525
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 20: loss 9.8125, lr 6.00e-06, 139.7k tok/s, FP8: False
iter 40: loss 9.5085, lr 1.20e-05, 144.9k tok/s, FP8: False
iter 60: loss 9.2443, lr 1.80e-05, 144.9k tok/s, FP8: False
iter 80: loss 9.0511, lr 2.40e-05, 144.5k tok/s, FP8: False
iter 100: loss 8.8182, lr 3.00e-05, 144.2k tok/s, FP8: True
iter 120: loss 8.3949, lr 3.60e-05, 163.0k tok/s, FP8: True
iter 140: loss 8.0966, lr 4.20e-05, 162.8k tok/s, FP8: True
iter 160: loss 7.7869, lr 4.80e-05, 162.5k tok/s, FP8: True
iter 180: loss 7.5582, lr 5.40e-05, 162.7k tok/s, FP8: True
iter 200: loss 7.2685, lr 6.00e-05, 162.8k tok/s, FP8: True
Step 200: val loss 6.8481
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 220: loss 6.8991, lr 6.60e-05, 155.7k tok/s, FP8: True
iter 240: loss 6.4105, lr 7.20e-05, 162.7k tok/s, FP8: True
iter 260: loss 6.1449, lr 7.80e-05, 162.5k tok/s, FP8: True
iter 280: loss 5.8949, lr 8.40e-05, 162.7k tok/s, FP8: True
iter 300: loss 5.7301, lr 9.00e-05, 162.5k tok/s, FP8: True
iter 320: loss 5.4801, lr 9.60e-05, 162.7k tok/s, FP8: True
iter 340: loss 5.1974, lr 1.02e-04, 162.7k tok/s, FP8: True
iter 360: loss 5.2358, lr 1.08e-04, 162.4k tok/s, FP8: True
iter 380: loss 5.0071, lr 1.14e-04, 162.6k tok/s, FP8: True
iter 400: loss 5.0241, lr 1.20e-04, 162.4k tok/s, FP8: True
Step 400: val loss 4.0019
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 420: loss 4.8821, lr 1.26e-04, 155.7k tok/s, FP8: True
iter 440: loss 4.7614, lr 1.32e-04, 162.6k tok/s, FP8: True
iter 460: loss 4.5980, lr 1.38e-04, 162.5k tok/s, FP8: True
iter 480: loss 4.5014, lr 1.44e-04, 162.5k tok/s, FP8: True
iter 500: loss 4.5639, lr 1.50e-04, 162.4k tok/s, FP8: True
Saving checkpoint to checkpoints_fp8_optimized/checkpoint_500_fp8_optimized.pt
iter 520: loss 4.4659, lr 1.56e-04, 161.7k tok/s, FP8: True
iter 540: loss 4.3473, lr 1.62e-04, 162.7k tok/s, FP8: True
iter 560: loss 4.2905, lr 1.68e-04, 162.5k tok/s, FP8: True
iter 580: loss 4.2656, lr 1.74e-04, 163.6k tok/s, FP8: True
iter 600: loss 4.2472, lr 1.80e-04, 162.6k tok/s, FP8: True
Step 600: val loss 3.0737
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt



TEST 7!! 
 python train_fp8_optimized_global.py --data_dir /workspace/yxanul_v1/mixed_6b/train.bin --vocab_size 32768 --n_layer 38 --n_head 8 --n_kv_heads 2 --n_embd 512 --opt sophia --sophia_lr 6e-4 --sophia_betas 0.965 0.99 --sophia_rho 0.05 --sophia_weight_decay 0.2 --sophia_k 10 --seed 1337 --batch_size 12 

  SophiaG lr=0.0006, betas=(0.965, 0.99), rho=0.05, wd=0.2, k=10
Batch size: 12
Gradient accumulation: 22
Effective batch size: 264
==================================================

iter 0: loss 10.4972, lr 0.00e+00, 122.5k tok/s, FP8: False
Step 0: val loss 10.4475
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 20: loss 9.8551, lr 6.00e-06, 124.0k tok/s, FP8: False
iter 40: loss 9.5597, lr 1.20e-05, 128.3k tok/s, FP8: False
iter 60: loss 9.3295, lr 1.80e-05, 128.2k tok/s, FP8: False
iter 80: loss 9.1193, lr 2.40e-05, 128.2k tok/s, FP8: False
iter 100: loss 8.8892, lr 3.00e-05, 128.6k tok/s, FP8: True
iter 120: loss 8.5028, lr 3.60e-05, 142.8k tok/s, FP8: True
iter 140: loss 8.1562, lr 4.20e-05, 142.8k tok/s, FP8: True
iter 160: loss 7.8452, lr 4.80e-05, 143.0k tok/s, FP8: True
iter 180: loss 7.6094, lr 5.40e-05, 143.3k tok/s, FP8: True
iter 200: loss 7.3267, lr 6.00e-05, 143.1k tok/s, FP8: True
Step 200: val loss 6.9506
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 220: loss 6.9574, lr 6.60e-05, 137.3k tok/s, FP8: True
iter 240: loss 6.4801, lr 7.20e-05, 143.0k tok/s, FP8: True
iter 260: loss 6.2131, lr 7.80e-05, 143.4k tok/s, FP8: True
iter 280: loss 5.9489, lr 8.40e-05, 143.5k tok/s, FP8: True
iter 300: loss 5.7841, lr 9.00e-05, 143.5k tok/s, FP8: True
iter 320: loss 5.5412, lr 9.60e-05, 143.1k tok/s, FP8: True
iter 340: loss 5.2500, lr 1.02e-04, 143.1k tok/s, FP8: True
iter 360: loss 5.2741, lr 1.08e-04, 142.8k tok/s, FP8: True
iter 380: loss 5.0440, lr 1.14e-04, 143.1k tok/s, FP8: True
iter 400: loss 5.0497, lr 1.20e-04, 143.4k tok/s, FP8: True
Step 400: val loss 4.0469
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 420: loss 4.9020, lr 1.26e-04, 137.2k tok/s, FP8: True
iter 440: loss 4.7798, lr 1.32e-04, 142.0k tok/s, FP8: True
iter 460: loss 4.6186, lr 1.38e-04, 142.1k tok/s, FP8: True
iter 480: loss 4.5043, lr 1.44e-04, 142.5k tok/s, FP8: True
iter 500: loss 4.5696, lr 1.50e-04, 142.7k tok/s, FP8: True
Saving checkpoint to checkpoints_fp8_optimized/checkpoint_500_fp8_optimized.pt
iter 520: loss 4.4616, lr 1.56e-04, 141.6k tok/s, FP8: True
iter 540: loss 4.3352, lr 1.62e-04, 142.8k tok/s, FP8: True
iter 560: loss 4.2734, lr 1.68e-04, 143.4k tok/s, FP8: True
iter 580: loss 4.2372, lr 1.74e-04, 143.2k tok/s, FP8: True
iter 600: loss 4.2138, lr 1.80e-04, 143.2k tok/s, FP8: True
Step 600: val loss 3.0353
Saving checkpoint to checkpoints_fp8_optimized/best_model_fp8_optimized.pt
iter 620: loss 4.0071, lr 1.86e-04, 137.0k tok/s, FP8: True