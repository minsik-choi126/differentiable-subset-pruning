# Experimental Setup Fixes - Paper Alignment & Numerical Stability

## 문제점 요약

### 1. 논문 Section 4.3과 불일치 ❌

**논문 (Li et al., TACL 2021 Section 4.3)**:
> **Pipelined pruning.** In this setting, the model is trained or fine-tuned on the target task **(3 epochs for BERT)** before being pruned. We learn the head importance weights for pipelined DSP for **one additional epoch**, and then select the top-k heads based on the learned importance. The model is fine-tuned afterwards with the pruned head set until convergence.

> **Joint pruning.** In this setting, the model is trained or fine-tuned for the **same number of epochs as pipelined pruning** while sparsity-enforcing regularization is applied.

**이전 구현**:
```python
# Joint DSP
num_epochs=2  # ❌ 논문: 3 epochs

# Pipelined DSP
num_epochs_base=1      # ❌ 논문: 3 epochs base training
num_epochs_score=1     # ✅ 논문: 1 additional epoch
num_epochs_ft=2        # ❌ 논문: until convergence (~3 epochs)
```

### 2. 수치 안정성 문제 (NaN Loss) 💥

**증상**:
- Epoch 2에서 loss가 NaN으로 폭발
- Accuracy가 85% → 49% (random level)로 하락
- 이상한 mask 패턴 (Layer 12이 완전히 pruning됨)

**근본 원인**:
1. **Final temperature가 너무 낮음**: 0.01 (should be 0.1)
   - `exp(w/0.01)` 계산 시 underflow 발생
   - Gumbel-Softmax가 불안정해짐
   - **Note**: Initial temperature 1000은 논문 Appendix A에서 명시한 올바른 값!

2. **Float64/Float32 타입 불일치**:
   - w 파라미터: `float64` (`.double()`)
   - BERT 모델: `float32`
   - Type casting overhead 및 정밀도 문제

3. **Gradient clipping 누락**:
   - w 파라미터의 gradient가 폭발할 수 있음

---

## 적용된 수정사항 ✅

### 1. 논문 Epoch 설정 반영

```python
# Joint DSP
num_epochs = 3  # 논문: "same number of epochs as pipelined pruning"

# Pipelined DSP
num_epochs_base = 3   # 논문: "3 epochs for BERT"
num_epochs_score = 1  # 논문: "one additional epoch"
num_epochs_ft = 3     # 논문: "until convergence" (BERT SST-2 표준)
```

**Compute Budget 비교**:
| 방법 | Base | w 학습 | Fine-tune | Total BERT 학습 |
|------|------|--------|-----------|----------------|
| **Pipelined** | 3 epochs | 1 epoch (BERT frozen) | 3 epochs | **6 epochs** |
| **Joint** | - | - | 3 epochs (w+BERT) | **3 epochs** |

**Note**: Pipelined의 w 학습 epoch은 BERT가 freeze되므로, 실제 BERT 학습 compute는 6 epochs.

### 2. Temperature 범위 수정 (논문 Appendix A 기준)

**논문 Appendix A - Table 2**:
- τini (initial temperature): **1000**
- τend (final temperature): **0.1**
- lr for wh: **0.5**
- Ncooldown: **25000**

**Before (Original Code)**:
```python
TemperatureScheduler(
    initial_temperature=1000.0,  # ✅ Correct (matches paper!)
    final_temperature=1e-2,      # ❌ Wrong (0.01, should be 0.1)
)
```

**After (Fixed)**:
```python
TemperatureScheduler(
    initial_temperature=1000.0,  # ✅ Paper Appendix A: τini = 1000
    final_temperature=0.1,       # ✅ Paper Appendix A: τend = 0.1
)
```

**효과**:
- Final temperature를 0.01 → 0.1로 수정 (10배 증가)
- `exp(w/0.01)`의 underflow 문제 해결
- `exp(w/0.1)`은 안정적인 범위
- **Initial temperature 1000은 논문에서 명시한 올바른 값!**

### 3. Dtype 일관성 (Float32)

**Before**:
```python
w = nn.Parameter(torch.randn(...).double())  # ❌ float64
p = torch.zeros(...).double()                # ❌ float64
soft_mask = gumbel_soft_top_k(...).double()  # ❌ float64
```

**After**:
```python
w = nn.Parameter(torch.randn(...))           # ✅ float32
p = torch.zeros(...)                         # ✅ float32
soft_mask = gumbel_soft_top_k(...)           # ✅ float32
```

**효과**:
- BERT 모델 dtype과 일치 (float32)
- Type casting overhead 제거
- 수치 정밀도 일관성 향상

### 4. Gradient Clipping 추가

**Before**:
```python
loss.backward()
optimizer_w.step()  # ❌ gradient clipping 없음
```

**After**:
```python
loss.backward()
torch.nn.utils.clip_grad_norm_([w], 1.0)  # ✅ gradient 폭발 방지
optimizer_w.step()
```

**효과**:
- w 파라미터 gradient 폭발 방지
- 학습 안정성 향상

---

## 수정 전후 비교

### Epoch Configuration

| Component | Before | After (Paper) | 변경 이유 |
|-----------|--------|---------------|----------|
| **Joint DSP epochs** | 2 | **3** | 논문 Section 4.3 명시 |
| **Pipelined base** | 1 | **3** | "3 epochs for BERT" |
| **Pipelined w learning** | 1 | 1 | "one additional epoch" ✅ |
| **Pipelined fine-tune** | 2 | **3** | "until convergence" |

### Numerical Stability

| Parameter | Before | After | 변경 이유 |
|-----------|--------|-------|----------|
| **Initial temp** | 1000.0 | **1000.0** ✅ | 논문 Appendix A 일치 |
| **Final temp** | 0.01 ❌ | **0.1** ✅ | 논문 Appendix A 일치 (Underflow 방지) |
| **w dtype** | float64 | **float32** | BERT와 일치 |
| **Gradient clip** | ❌ 없음 | **✅ 1.0** | 안정성 향상 |

---

## 예상 효과

### 1. 논문과 일치하는 공정한 비교

- Pipelined DSP가 논문에서 의도한 대로 "3 epochs base + 1 epoch w learning" 수행
- Joint DSP와 동일한 기준 (3 epochs)으로 비교 가능
- Compute budget 명확히 정의됨

### 2. 안정적인 학습

- ✅ NaN loss 방지
- ✅ Accuracy collapse 방지
- ✅ 합리적인 mask 패턴 (모든 layer에 분산)

### 3. 재현 가능한 결과

- 논문 결과와 유사한 패턴 기대:
  ```
  Joint DSP (End-to-end):
    - 10% pruning: ~90% accuracy
    - 30% pruning: ~89% accuracy
    - 50% pruning: ~86% accuracy

  Pipelined DSP (Fair comparison):
    - 10% pruning: ~89.5% accuracy
    - 30% pruning: ~88.5% accuracy
    - 50% pruning: ~85% accuracy
  ```

---

## 체크리스트

수정사항이 올바르게 적용되었는지 확인:

- [x] **Joint DSP**: `num_epochs=3`
- [x] **Pipelined base**: `num_epochs_base=3`
- [x] **Pipelined w learning**: `num_epochs_score=1`
- [x] **Pipelined fine-tune**: `num_epochs_ft=3`
- [x] **Temperature initial**: `1000.0` (논문 Appendix A)
- [x] **Temperature final**: `0.1`
- [x] **w dtype**: `float32` (no `.double()`)
- [x] **p dtype**: `float32` (no `.double()`)
- [x] **Gradient clipping**: `clip_grad_norm_([w], 1.0)`

---

## Colab 빠른 테스트용 설정

논문 설정은 시간이 오래 걸리므로, Colab 테스트용으로 축소 버전 제공:

```python
if IN_COLAB:
    # 빠른 테스트 (30분)
    num_epochs = 1
    num_epochs_base = 1
    num_epochs_ft = 1
else:
    # 논문 설정 (3시간)
    num_epochs = 3
    num_epochs_base = 3
    num_epochs_ft = 3
```

**Note**: 논문과 정확히 일치하는 결과를 얻으려면 `IN_COLAB=False` 설정 사용!

---

## 참고 자료

- **논문**: Li et al. "Differentiable Subset Pruning of Transformer Heads" (TACL 2021)
- **Section**: 4.3 Experimental Setup
- **Dataset**: SST-2 (Stanford Sentiment Treebank)
- **Model**: BERT-base-uncased

---

**요약**: 이제 구현이 논문 Section 4.3의 실험 설정을 충실히 반영하며, 수치 안정성 문제도 해결되었습니다! 🎉
