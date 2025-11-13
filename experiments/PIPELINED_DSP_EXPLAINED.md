# Pipelined DSP vs Michel et al. - 핵심 차이점

## 🎯 핵심 개념

### ❌ 잘못된 이해 (제가 처음 구현한 것)

```
Pipelined DSP = Michel et al. 방식 + Fine-tuning
```

### ✅ 올바른 이해

```
Pipelined DSP = DSP score를 1 epoch만 학습 + Top-K mask + Fine-tuning
```

## 📊 세 가지 방법 비교

### 1. Michel et al. (2019) - Gradient-based Importance

```python
# Stage 1: Train full model
model = train(model, data)

# Stage 2: Compute importance
head_mask = torch.ones(n_layers, n_heads)
head_mask.requires_grad = True

for batch in data:
    loss = model(batch, head_mask=head_mask).loss
    loss.backward()
    importance += head_mask.grad.abs()  # |∂L/∂mask|

# Stage 3: Prune + Fine-tune
mask = top_k(importance, k)
model = fine_tune(model, data, mask=mask)
```

**핵심:**
- Importance = Gradient의 절대값 `|∂L/∂mask|`
- One-pass 계산 (모델 forward-backward 1회)
- Importance 자체는 학습되지 않음

---

### 2. Joint DSP (Li et al., 2021) - End-to-End

```python
# Learnable importance weights
w = nn.Parameter(torch.randn(n_layers, n_heads))

# Training loop
for epoch in epochs:
    for batch in data:
        # Differentiable mask
        mask = gumbel_soft_top_k(w, k, temperature)

        # Forward
        loss = model(batch, head_mask=mask).loss

        # Backward (w도 학습!)
        loss.backward()
        optimizer.step()  # w와 BERT weights 모두 업데이트

# Final mask
final_mask = top_k(w, k)
```

**핵심:**
- `w`가 learnable parameter
- Task loss로 직접 w 학습
- End-to-end differentiable
- Temperature annealing으로 soft → hard

---

### 3. Pipelined DSP (올바른 구현) ⭐

```python
# Stage 1: Base fine-tuning (optional)
model = train(model, data)

# Stage 2: DSP score (w)를 1 epoch만 학습
w = nn.Parameter(torch.randn(n_layers, n_heads))

# BERT freeze, w만 학습
for param in model.parameters():
    param.requires_grad = False

for batch in data:  # 1 epoch만!
    mask = gumbel_soft_top_k(w, k, temperature)
    loss = model(batch, head_mask=mask).loss
    loss.backward()
    w_optimizer.step()  # w만 업데이트

# Stage 3: Top-K mask 생성
final_mask = top_k(w, k)

# Stage 4: Mask 고정하고 BERT fine-tune
for param in model.parameters():
    param.requires_grad = True

model = fine_tune(model, data, mask=final_mask)
```

**핵심:**
- `w`를 1 epoch만 학습 (BERT freeze)
- Michel et al.과 **동일한 compute budget** (gradient 계산 횟수)
- DSP의 differentiable top-k 사용
- 학습된 w로 mask 생성 후 고정

## 🔍 왜 이렇게 하는가?

### Compute Budget 비교

| 방법 | Importance 계산 | Fine-tuning | 총 Epoch |
|------|----------------|-------------|----------|
| Michel et al. | 1 epoch (gradient) | 3 epochs | 4 |
| Joint DSP | - | 3 epochs (w+BERT) | 3 |
| Pipelined DSP | 1 epoch (w만) | 3 epochs (BERT) | 4 |

**Pipelined DSP의 목적:**
- Michel et al.과 **공정한 비교** (동일한 compute)
- DSP 기법의 효과만 isolate해서 측정
- "DSP score가 gradient보다 나은가?" 검증

## 💻 코드 비교

### 잘못된 구현 (제가 처음 한 것)

```python
def run_pipelined_dsp_WRONG():
    # Stage 1: Base training
    model = train(model)

    # Stage 2: Michel et al. importance ❌
    importance = compute_head_importance(model, data)
    # → |∂L/∂mask| 계산 (gradient 기반)

    # Stage 3: Mask
    mask = top_k(importance, k)

    # Stage 4: Fine-tune
    model = fine_tune(model, data, mask)
```

### 올바른 구현 ✅

```python
def run_pipelined_dsp_CORRECT():
    # Stage 1: Base training
    model = train(model)

    # Stage 2: DSP score (w) 학습 ✅
    w = nn.Parameter(torch.randn(n_layers, n_heads))

    # BERT freeze
    for p in model.parameters():
        p.requires_grad = False

    # w만 1 epoch 학습
    for batch in data_1_epoch:
        mask = gumbel_soft_top_k(w, k, temperature)
        loss = model(batch, head_mask=mask).loss
        loss.backward()
        w_optimizer.step()  # w만 업데이트

    # Stage 3: Mask 생성
    mask = top_k(w, k)

    # Stage 4: BERT fine-tune (mask 고정)
    for p in model.parameters():
        p.requires_grad = True

    model = fine_tune(model, data, mask)
```

## 📈 예상 결과

### Michel et al. vs Pipelined DSP

만약 Pipelined DSP가 더 좋다면:
```
Pruning 30%:
- Michel et al.:    87.5% accuracy
- Pipelined DSP:    88.2% accuracy ← DSP score가 더 효과적!
```

**의미:**
- Gradient 기반 importance보다
- DSP의 learnable importance가 더 좋은 head를 선택

### Joint vs Pipelined DSP

일반적으로 Joint DSP가 더 좋을 것:
```
Pruning 50%:
- Pipelined DSP:   85.1% accuracy
- Joint DSP:       86.5% accuracy ← End-to-end의 힘!
```

**의미:**
- End-to-end 최적화가 더 효과적
- 하지만 compute budget은 다름 (3 vs 4 epoch)

## 🎓 논문에서의 설명

Li et al. (TACL 2021) 논문에서:

> "We compare our method with the importance-based pruning method of Michel et al. (2019). To ensure a fair comparison in terms of compute budget, we also evaluate a pipelined variant where we first learn the importance scores for one epoch, then select the top-k heads and fine-tune."

**핵심:**
- Fair comparison을 위해 compute budget 맞춤
- Importance score = DSP의 learnable weights (w)
- 1 epoch 학습 = Michel et al.과 동일한 gradient 계산 횟수

## 🔧 구현 체크리스트

Pipelined DSP를 올바르게 구현했는지 확인:

- [ ] Stage 2에서 `w = nn.Parameter()` 사용
- [ ] Stage 2에서 BERT parameters는 freeze
- [ ] Stage 2에서 `gumbel_soft_top_k()` 사용
- [ ] Stage 2는 **정확히 1 epoch**
- [ ] Stage 3에서 `w`로부터 top-k mask 생성
- [ ] Stage 4에서 mask는 `requires_grad=False`
- [ ] Stage 4에서 BERT만 fine-tune

## 📊 정리

| 항목 | Michel et al. | Pipelined DSP | Joint DSP |
|------|--------------|---------------|-----------|
| **Importance** | Gradient | Learned (w) | Learned (w) |
| **학습 방식** | One-pass | 1 epoch (w만) | Multi-epoch (w+BERT) |
| **Top-K 방법** | Hard | Hard | Soft→Hard |
| **BERT 학습** | Initial + FT | Initial + FT | End-to-end |
| **Compute** | 4 epoch | 4 epoch | 3 epoch |
| **목적** | Baseline | Fair comparison | Best performance |

---

**결론:**
Pipelined DSP는 Michel et al.과 공정하게 비교하기 위한 방법으로,
**DSP의 learnable importance가 gradient보다 나은지** 검증합니다!
