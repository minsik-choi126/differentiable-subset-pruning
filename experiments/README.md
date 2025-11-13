# SST-2 Joint vs Pipelined DSP Experiments

이 디렉토리는 SST-2 task에서 Joint DSP와 Pipelined DSP를 비교하는 실험을 포함합니다.

## 실험 개요

### 모델
- **BERT-base-uncased**: 12 layers × 12 heads = 144 total heads

### 태스크
- **SST-2** (Stanford Sentiment Treebank): Binary sentiment classification

### Pruning Ratios
- **10%**: 144 → 130 heads (14 heads pruned)
- **30%**: 144 → 101 heads (43 heads pruned)
- **50%**: 144 → 72 heads (72 heads pruned)

### 방법 비교

#### 1. Joint DSP (End-to-End)
```
┌─────────────────────────────────────────┐
│ 초기화: w = learnable importance weights │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Training Loop:                          │
│  1. mask = gumbel_soft_top_k(w, k)     │
│  2. outputs = BERT(input, mask)         │
│  3. loss = CrossEntropy(outputs, label) │
│  4. loss.backward() → w 학습!           │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 최종: hard_mask = top_k(w, k)          │
└─────────────────────────────────────────┘
```

**특징:**
- ✅ End-to-end differentiable
- ✅ Task loss가 직접 head selection 학습
- ✅ Temperature annealing으로 soft→hard 전환
- ⚠️ 학습 중 soft mask 사용 (약간의 overhead)

#### 2. Pipelined DSP (Two-Stage)
```
┌─────────────────────────────────────────┐
│ Stage 1: Base model fine-tuning        │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Stage 2: Importance = |∂loss/∂mask|    │
│         (gradient-based)                │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Stage 3: mask = top_k(importance, k)   │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Stage 4: Fine-tune with fixed mask     │
└─────────────────────────────────────────┘
```

**특징:**
- ✅ 이해하기 쉬운 pipeline
- ✅ 각 stage가 독립적
- ✅ Fine-tuning 중 hard mask 사용
- ⚠️ Two-stage (importance 계산이 별도)

## 설치

### 1. 환경 설정

```bash
# Python 3.8+ 권장
conda create -n dsp python=3.8
conda activate dsp
```

### 2. 필수 패키지 설치

```bash
pip install torch torchvision torchaudio
pip install transformers datasets
pip install numpy tqdm
```

또는:

```bash
pip install -r requirements.txt
```

## 실행 방법

### Quick Start (빠른 테스트)

작은 샘플로 빠르게 테스트:

```bash
cd experiments
python sst2_joint_vs_pipelined.py
```

스크립트 내에서 `max_train_samples`를 수정하여 샘플 수 조절 가능:

```python
# sst2_joint_vs_pipelined.py 내에서
max_train_samples = 5000  # 5000개만 사용 (빠른 테스트)
max_train_samples = None  # 전체 데이터 사용 (full experiment)
```

### Full Experiment

전체 데이터로 실행 (시간이 오래 걸림):

```bash
python sst2_joint_vs_pipelined.py
```

**예상 실행 시간** (GPU: V100 기준):
- Quick test (5000 samples): ~30분
- Full experiment: ~3-4시간

### GPU 메모리 요구사항

- **Minimum**: 8GB (batch_size=16)
- **Recommended**: 16GB (batch_size=32)
- **Full**: 24GB+ (batch_size=64)

배치 사이즈 조정:

```python
# 스크립트 내에서
batch_size = 16  # GPU 메모리 부족시 줄이기
```

## 출력 결과

### 1. 콘솔 출력

실행 중 실시간 로그:

```
================================================================================
Joint DSP - Pruning Ratio: 10%
================================================================================
Total heads: 144
Keeping: 130 (90.3%)
Pruning: 14 (9.7%)

Epoch 1/3
Training: 100%|████████| 2104/2104 [05:23<00:00, 6.51it/s, loss=0.3421, temp=8.91e-01]

[Step 200] loss=0.3421, eval_loss=0.2891, acc=89.45%, best=89.45%, temp=8.91e-01, sparsity=9.7%

...

[Joint DSP] Final Results:
  Pruning ratio: 10%
  Keep heads: 130/144
  Final accuracy: 90.23%
  Best accuracy: 90.56%
```

### 2. 저장된 결과

`experiments/results/sst2_joint_vs_pipelined.json`:

```json
{
  "config": {
    "task": "sst2",
    "model": "bert-base-uncased",
    "batch_size": 32,
    "pruning_ratios": [0.1, 0.3, 0.5]
  },
  "joint_dsp": [
    {
      "method": "joint_dsp",
      "pruning_ratio": 0.1,
      "keep_heads": 130,
      "total_heads": 144,
      "final_accuracy": 0.9023,
      "best_accuracy": 0.9056,
      "history": {...}
    },
    ...
  ],
  "pipelined_dsp": [...]
}
```

### 3. 최종 요약

```
================================================================================
FINAL RESULTS SUMMARY
================================================================================

Task: SST-2
Model: BERT-base-uncased (12 layers × 12 heads = 144 total heads)

--------------------------------------------------------------------------------
Joint DSP Results:
--------------------------------------------------------------------------------
  Prune %     Keep   Final Acc     Best Acc
--------------------------------------------------------------------------------
      10%  130/144       90.23%       90.56%
      30%  101/144       89.12%       89.67%
      50%   72/144       86.45%       87.21%

--------------------------------------------------------------------------------
Pipelined DSP Results:
--------------------------------------------------------------------------------
  Prune %     Keep     Base  Before FT   Final Acc     Best Acc
--------------------------------------------------------------------------------
      10%  130/144   91.28%     90.45%       90.78%       90.95%
      30%  101/144   91.28%     88.23%       89.34%       89.56%
      50%   72/144   91.28%     84.12%       86.89%       87.34%
```

## 결과 분석

### 예상 결과

일반적으로 다음과 같은 패턴이 관찰됩니다:

1. **낮은 pruning ratio (10%)**:
   - 두 방법 모두 base 성능의 ~99% 유지
   - Pipelined DSP가 약간 더 안정적

2. **중간 pruning ratio (30%)**:
   - Joint DSP가 end-to-end 최적화로 약간 우수
   - Pipelined DSP는 importance 계산의 정확도에 의존

3. **높은 pruning ratio (50%)**:
   - Joint DSP의 장점이 더 명확히 드러남
   - Hard constraint가 효과적으로 작동

### 결과 시각화

결과를 시각화하려면:

```python
import json
import matplotlib.pyplot as plt

with open('experiments/results/sst2_joint_vs_pipelined.json') as f:
    results = json.load(f)

ratios = [r['pruning_ratio'] * 100 for r in results['joint_dsp']]
joint_acc = [r['final_accuracy'] * 100 for r in results['joint_dsp']]
pipelined_acc = [r['final_accuracy'] * 100 for r in results['pipelined_dsp']]

plt.figure(figsize=(10, 6))
plt.plot(ratios, joint_acc, 'o-', label='Joint DSP', linewidth=2)
plt.plot(ratios, pipelined_acc, 's-', label='Pipelined DSP', linewidth=2)
plt.xlabel('Pruning Ratio (%)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('SST-2: Joint vs Pipelined DSP', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig('sst2_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 문제 해결

### CUDA Out of Memory

```bash
# 배치 사이즈 줄이기
batch_size = 16  # 또는 8
```

### Slow Training

```bash
# 샘플 수 제한
max_train_samples = 5000

# Epoch 수 줄이기
num_epochs = 2  # Joint DSP
num_epochs_ft = 2  # Pipelined DSP
```

### Dataset Download Error

```bash
# Hugging Face datasets 캐시 확인
export HF_DATASETS_CACHE="./cache"
```

## 커스터마이징

### 다른 Pruning Ratio 시도

```python
pruning_ratios = [0.2, 0.4, 0.6, 0.8]  # 20%, 40%, 60%, 80%
```

### Hyperparameter 조정

#### Joint DSP:

```python
# Temperature schedule
initial_temperature = 1000.0  # 더 soft한 시작
final_temperature = 1e-2      # 더 hard한 종료

# Learning rates
model_lr = 2e-5    # BERT weights
importance_lr = 0.5  # Importance weights (w)
```

#### Pipelined DSP:

```python
# Stage별 epoch 조정
num_epochs_base = 2   # Base fine-tuning
num_epochs_ft = 4     # Pruned fine-tuning
```

### STE vs Gumbel-Softmax

Joint DSP에서 STE 사용:

```python
result = run_joint_dsp(
    pruning_ratio=ratio,
    train_loader=train_loader,
    eval_loader=eval_loader,
    use_ste=True,  # Straight-Through Estimator
)
```

## 참고 자료

1. **Differentiable Subset Pruning of Transformer Heads**
   - Li et al., TACL 2021
   - https://arxiv.org/abs/2108.04657

2. **Are Sixteen Heads Really Better than One?**
   - Michel et al., NeurIPS 2019
   - https://arxiv.org/abs/1905.10650

3. **BERT: Pre-training of Deep Bidirectional Transformers**
   - Devlin et al., NAACL 2019
   - https://arxiv.org/abs/1810.04805

## 라이선스

본 실험 코드는 연구 및 교육 목적으로 자유롭게 사용 가능합니다.
