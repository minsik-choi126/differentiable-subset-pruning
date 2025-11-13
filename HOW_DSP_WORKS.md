# DSP (Differentiable Subset Pruning) - 정확한 작동 원리

## 핵심 개념

DSP는 **매 forward pass마다** importance weights를 사용해서 동적으로 mask를 생성하고, attention probabilities에 곱하는 방식으로 pruning을 학습합니다.

## 전체 흐름

```
[초기화]
┌─────────────────────────────────────────────────────┐
│ 1. 모델에 learnable importance weights 추가         │
│    w = nn.Parameter([n_layers, n_heads])            │
│    shape: [12, 12] for BERT-base                    │
└─────────────────────────────────────────────────────┘

[매 Training Step]
┌─────────────────────────────────────────────────────┐
│ 2. Temperature 계산 (annealing)                     │
│    T(t) = exp(log(T₀) - t/τ * log(T₀/Tₓ))         │
│    1000 → ... → 1e-8                                │
└─────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────┐
│ 3. Mask 생성 (differentiable top-k)                │
│    mask = gumbel_soft_top_k(w, k=48, temp=T)       │
│    또는                                              │
│    mask = STEFunction.apply(w, k=48)               │
└─────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────┐
│ 4. Attention에 mask 적용                            │
│    attention_probs = softmax(QK^T/√d)              │
│    attention_probs *= mask  ← 여기서 pruning!      │
│    context = attention_probs @ V                    │
└─────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────┐
│ 5. Task loss 계산 및 backprop                       │
│    loss = CrossEntropy(logits, labels)             │
│    loss.backward()  → w에 gradient 흐름!           │
└─────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────┐
│ 6. Optimizer로 w 업데이트                           │
│    w ← w - lr * ∂loss/∂w                           │
│    (중요한 head는 w↑, 불필요한 head는 w↓)          │
└─────────────────────────────────────────────────────┘

[Training 완료 후]
┌─────────────────────────────────────────────────────┐
│ 7. 학습된 w로 최종 mask 생성                        │
│    final_mask = top_k(w, k=48)                     │
│    → 상위 48개 head만 1, 나머지 0                   │
└─────────────────────────────────────────────────────┘
```

## 상세 단계별 설명

### Step 1: 초기화

```python
# GatedBertForSequenceClassification.__init__()
self.w = nn.Parameter(torch.empty([12, 12]).double())  # learnable!
nn.init.xavier_uniform_(self.w)

# 예시:
# w = [[0.23, -0.15, 0.87, ...],  # layer 0의 각 head 중요도
#      [0.45,  0.92, -0.34, ...],  # layer 1의 각 head 중요도
#      ...]
```

**핵심**: `w`는 학습 가능한 파라미터입니다! 모델 weights와 함께 학습됩니다.

### Step 2: Temperature Annealing

```python
# DSPTrainer - 매 step마다 호출
if self.annealing and step <= cooldown_steps:
    temperature = np.exp(
        np.log(1000.0) -
        step / 25000 *
        (np.log(1000.0) - np.log(1e-8))
    )
else:
    temperature = 1e-8

# Step별 temperature 변화:
# Step     0: T = 1000.0    (매우 soft)
# Step  5000: T = 31.62
# Step 10000: T = 1.0
# Step 15000: T = 0.0316
# Step 25000: T = 1e-8      (거의 hard)
```

**목적**: 처음엔 soft selection으로 탐색, 나중엔 hard selection으로 수렴

### Step 3: Mask 생성 (핵심!)

#### 방법 A: Gumbel-Softmax (기본)

```python
# GatedBertForSequenceClassification.forward()
if self.use_dsp:
    # w: [12, 12] → flatten → [144]
    head_mask = gumbel_soft_top_k(
        self.w.view(-1),      # [144] 모든 head의 importance
        self.num_of_heads,     # 48 (keep할 개수)
        self.temperature       # 현재 temperature
    ).view_as(self.w)         # → [12, 12]로 reshape
```

**Gumbel-Softmax 내부 동작:**

```python
def gumbel_soft_top_k(w, k, temperature):
    # 1. Gumbel noise 추가
    u = torch.rand_like(w)  # [0, 1] uniform
    gumbel = -log(-log(u))   # Gumbel(0, 1)
    r = w + gumbel           # importance + randomness

    # 2. 첫 번째 선택 (가장 중요한)
    p[0] = softmax(r / temperature)  # soft selection

    # 3. 두 번째 선택 (이미 선택된 것 제외)
    r = r + log(1 - p[0])    # 첫 번째 확률 제거
    p[1] = softmax(r / temperature)

    # 4-48. 반복...
    # ...

    # 최종: k개 선택의 확률 합산
    return p.sum(0)  # 각 head가 선택될 총 확률
```

**결과 예시 (T=10일 때):**
```
mask = [0.92, 0.03, 0.88, 0.05, 0.91, 0.02, ...]
       ↑soft!  거의0  선택됨  거의0  선택됨  거의0
```

**결과 예시 (T=0.001일 때):**
```
mask = [1.00, 0.00, 1.00, 0.00, 1.00, 0.00, ...]
       ↑거의 hard selection!
```

#### 방법 B: Straight-Through Estimator

```python
if self.use_ste:
    head_mask = STEFunction.apply(self.w.view(-1), 48)

# STEFunction 내부:
# Forward:  hard top-k selection (정확히 48개만 1)
# Backward: gradient 그대로 통과 (미분 가능)
```

### Step 4: Attention에 Mask 적용

```python
# GatedBertSelfAttention.forward()
# Attention 계산
query, key, value = self.query(x), self.key(x), self.value(x)
attention_scores = query @ key.T / sqrt(d_k)
attention_probs = softmax(attention_scores)  # [batch, n_heads, seq, seq]

# 🔥 여기서 pruning 발생!
if self.head_mask is not None:
    attention_probs = attention_probs * self.head_mask
    # head_mask shape: [1, n_heads, 1, 1] - broadcasting됨

# Context 계산
context = attention_probs @ value
```

**구체적 예시:**

```python
# Layer 3의 attention_probs (mask 적용 전)
# shape: [batch=32, n_heads=12, seq=128, seq=128]

# head_mask for layer 3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]
#                         ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
#                       head 0,1,2,3,4,5,6,7,8,9,10,11

# Mask 적용 (broadcasting)
attention_probs *= head_mask.view(1, 12, 1, 1)

# 결과:
# - Head 0,2,3,5,8,9,11: attention_probs 그대로 (×1)
# - Head 1,4,6,7,10:    attention_probs = 0 (×0) ← PRUNED!
```

**중요**:
- mask가 0인 head는 attention이 완전히 0이 됨
- 즉, 그 head는 출력에 전혀 기여하지 않음 = pruned!

### Step 5: Loss 계산 및 Backpropagation

```python
# Task loss
outputs = model(input_ids, labels=labels)
logits = outputs.logits
loss = CrossEntropyLoss()(logits, labels)

# Backward
loss.backward()

# Gradient flow:
# loss → logits → context → attention_probs → mask → w
#                                              ↑
#                            여기서 w가 gradient 받음!
```

**Gradient 흐름 예시:**

```
만약 head 5가 task에 중요하다면:
- head 5가 pruning되면 loss↑
- ∂loss/∂(head 5 mask) < 0  (mask↑하면 loss↓)
- ∂loss/∂w[layer, 5] < 0    (w↑하면 mask↑하면 loss↓)
- w[layer, 5] ← w[layer, 5] - lr * (음수) = w ↑

만약 head 7이 불필요하다면:
- head 7이 pruning되어도 loss 변화 없음
- ∂loss/∂w[layer, 7] ≈ 0
- w[layer, 7] 변화 없거나 감소
```

### Step 6: Optimizer Update

```python
# run_dsp.py - optimizer 설정
optimizer = AdamW([
    {'params': [p for n, p in model.named_parameters() if n != 'w'],
     'lr': 2e-5},           # 일반 BERT parameters
    {'params': [p for n, p in model.named_parameters() if n == 'w'],
     'lr': 0.5}             # importance weights (높은 lr!)
])

# Optimizer step
optimizer.step()  # w가 업데이트됨!
```

**핵심**: `w`는 일반 파라미터보다 **훨씬 높은 learning rate** (0.5 vs 2e-5)를 사용합니다!

### Step 7: 최종 Mask 생성

```python
# Training 완료 후
model.use_dsp = False  # DSP 끄기

# 학습된 w로 hard mask 생성
final_mask = convert_gate_to_mask(
    model.get_w(),      # 학습된 importance weights
    num_of_heads=48     # top-48 선택
)

# final_mask 예시:
# [[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1],  # layer 0
#  [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0],  # layer 1
#  ...]
# 정확히 48개의 1, 96개의 0

# 이 mask를 모델에 고정
model.apply_masks(final_mask)

# 이제 inference시 이 mask 사용 (w 학습 X)
```

## 두 가지 방법 비교

### Gumbel-Softmax DSP

```python
# 장점:
# - Fully differentiable (진짜 미분 가능)
# - Soft selection으로 smooth한 학습
# - Temperature로 soft→hard 점진적 전환

# 단점:
# - Forward pass에서 soft mask (0~1 연속값)
# - 실제 pruning 효과를 training 중 완전히 보기 어려움

# 사용 예:
trainer = DSPTrainer(
    model=model,
    num_of_heads=48,
    use_ste=False,
    annealing=True,
    initial_temperature=1000.0,
    final_temperature=1e-8,
    cooldown_steps=25000
)
```

### Straight-Through Estimator (STE)

```python
# 장점:
# - Forward에서 hard selection (정확히 48개)
# - Training 중에도 실제 pruning 효과 반영
# - 더 빠른 수렴

# 단점:
# - Biased gradient (수학적으로 정확하지 않음)
# - Forward와 backward가 불일치

# 사용 예:
trainer = DSPTrainer(
    model=model,
    num_of_heads=48,
    use_ste=True,
    # temperature 관련 설정 불필요
)
```

## 실제 코드 예시

### 전체 Training Loop

```python
# 1. Setup
model = GatedBertForSequenceClassification.from_pretrained('bert-base-uncased')
# 이때 model.w가 자동으로 초기화됨

# 2. Optimizer (w에 높은 lr)
optimizer = AdamW([
    {'params': [p for n, p in model.named_parameters() if n != 'w'],
     'lr': 2e-5},
    {'params': [p for n, p in model.named_parameters() if n == 'w'],
     'lr': 0.5},
])

# 3. Training loop
temp_scheduler = TemperatureScheduler(1000.0, 1e-8, 25000)

for step, batch in enumerate(dataloader):
    # 3a. Temperature 업데이트
    temperature = temp_scheduler.get_temperature(step)

    # 3b. DSP 활성화 (매 step마다!)
    model.apply_dsp(num_of_heads=48, temperature=temperature)

    # 3c. Forward (내부에서 mask 생성 & 적용)
    outputs = model(**batch)
    loss = outputs.loss

    # 3d. Backward (w에 gradient)
    loss.backward()

    # 3e. Update (w 업데이트!)
    optimizer.step()
    optimizer.zero_grad()

# 4. Training 완료 후
model.use_dsp = False
final_mask = convert_gate_to_mask(model.get_w(), num_of_heads=48)
model.apply_masks(final_mask)

# 5. Inference (고정된 mask 사용)
outputs = model(**test_batch)  # pruned model!
```

## 핵심 요약

1. **Learnable importance weights (`w`)**: 각 head의 중요도를 학습
2. **Differentiable top-k**: Gumbel-Softmax로 top-k 선택을 미분 가능하게
3. **Dynamic masking**: 매 forward마다 w로부터 mask 생성
4. **Attention pruning**: mask를 attention probabilities에 곱해서 pruning
5. **End-to-end learning**: Task loss로 w를 직접 학습
6. **Temperature annealing**: Soft→hard selection으로 점진적 수렴
7. **Final hard mask**: 학습 완료 후 top-k로 hard mask 생성

## 왜 이게 효과적인가?

```
기존 방법 (Michel et al.):
1. 전체 모델 학습
2. Importance 계산 (별도 과정)
3. 중요도 낮은 head 제거
4. (선택적) Fine-tuning
→ Two-stage, importance 계산이 별도

DSP:
1. Task 학습 + head selection 동시 학습
2. End-to-end differentiable
3. Hard constraint (정확히 k개)
→ One-stage, optimal subset 찾기
```

DSP는 "어떤 k개 head를 남겨야 task 성능이 최대인가?"를 **직접 최적화**합니다!
