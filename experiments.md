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