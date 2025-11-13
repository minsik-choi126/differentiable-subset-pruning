# Google Colab에서 DSP 실험 실행하기 🚀

## 빠른 시작 (3단계)

### 1️⃣ Colab에서 파일 열기

**방법 A: 직접 업로드**
1. [Google Colab](https://colab.research.google.com) 접속
2. `File > Upload notebook` 클릭
3. `sst2_dsp_experiments_colab.py` 업로드

**방법 B: GitHub에서 직접**
1. GitHub에서 파일 URL 복사
2. Colab에서 `File > Open notebook > GitHub` 탭
3. URL 붙여넣기

### 2️⃣ GPU 활성화

```
Runtime > Change runtime type > Hardware accelerator > GPU > Save
```

**추천:** T4 GPU (무료)

### 3️⃣ 실행

```python
Runtime > Run all
```

또는 `Ctrl+F9` (Windows) / `Cmd+F9` (Mac)

## 예상 실행 시간 ⏱️

| 설정 | GPU | 시간 |
|------|-----|------|
| Quick (5000 samples) | T4 | ~30분 |
| Full dataset | T4 | ~3시간 |
| Quick | V100 | ~15분 |
| Full | V100 | ~1.5시간 |

## Colab 코드 셀로 실행하기

Colab에서 `.py` 파일 대신 코드 셀로 실행하려면:

### 📝 셀 1: 설치 및 Import

```python
# 패키지 설치
!pip install -q transformers datasets torch

# Import
import os
import json
import random
import math
from typing import Dict, Tuple, List, Optional
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    default_data_collator,
    AdamW,
    get_linear_schedule_with_warmup,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"디바이스: {DEVICE}")
```

### 📝 셀 2: DSP 함수 정의

```python
# sst2_dsp_experiments_colab.py의 함수 정의 부분 복사
# (gumbel_soft_top_k, STEFunction, etc.)
```

### 📝 셀 3: 실험 실행

```python
# main() 함수 실행
main()
```

## 설정 커스터마이징 🔧

### 빠른 테스트 (5분)

```python
# main() 함수 안에서
pruning_ratios = [0.3]  # 1개 ratio만
max_train_samples = 1000  # 1000개만
num_epochs = 1  # 1 epoch만
```

### 중간 테스트 (30분)

```python
pruning_ratios = [0.1, 0.3, 0.5]  # 기본값
max_train_samples = 5000
num_epochs = 2
```

### Full 실험 (3시간)

```python
pruning_ratios = [0.1, 0.3, 0.5]
max_train_samples = None  # 전체 데이터
num_epochs = 3
```

## 메모리 최적화 💾

Colab 무료 버전 (12-15GB RAM):

```python
# 배치 사이즈 줄이기
batch_size = 16  # 기본값 32에서 줄임

# Gradient accumulation
# (더 작은 배치로 큰 배치 효과)
accumulation_steps = 2
```

## 결과 확인 📊

### 실시간 로그

```
================================================================================
Joint DSP - Pruning Ratio: 10%
================================================================================
전체 heads: 144
유지: 130 (90.3%)

Epoch 1/2
훈련: 100%|██████████| 157/157 [02:15<00:00]

[Step 200] loss=0.3421, eval_acc=89.45%, best=89.45%
```

### 최종 요약

```
================================================================================
📊 최종 결과 요약
================================================================================

Joint DSP:
--------------------------------------------------------------------------------
  Prune %   Final Acc     Best Acc
--------------------------------------------------------------------------------
      10%       90.23%       90.56%
      30%       89.12%       89.67%
      50%       86.45%       87.21%

Pipelined DSP:
--------------------------------------------------------------------------------
  Prune %     Base   Pruned    Final
--------------------------------------------------------------------------------
      10%    91.28%   90.45%   90.78%
      30%    91.28%   88.23%   89.34%
      50%    91.28%   84.12%   86.89%
```

### 결과 파일 다운로드

Colab 왼쪽 사이드바 > Files 탭에서:
- `sst2_results.json` 다운로드

```python
# 결과 읽기
import json

with open('sst2_results.json') as f:
    results = json.load(f)

print(json.dumps(results, indent=2))
```

## 시각화 📈

### 결과 그래프 그리기

```python
import matplotlib.pyplot as plt

with open('sst2_results.json') as f:
    results = json.load(f)

ratios = [r['pruning_ratio'] * 100 for r in results['joint_dsp']]
joint_acc = [r['final_accuracy'] * 100 for r in results['joint_dsp']]
pipelined_acc = [r['final_accuracy'] * 100 for r in results['pipelined_dsp']]

plt.figure(figsize=(10, 6))
plt.plot(ratios, joint_acc, 'o-', label='Joint DSP', linewidth=2, markersize=8)
plt.plot(ratios, pipelined_acc, 's-', label='Pipelined DSP', linewidth=2, markersize=8)
plt.xlabel('Pruning Ratio (%)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('SST-2: Joint vs Pipelined DSP', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(80, 95)

plt.tight_layout()
plt.savefig('sst2_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("그래프 저장됨: sst2_comparison.png")
```

## 문제 해결 🔧

### 1. CUDA Out of Memory

```python
# 해결책 1: 배치 사이즈 줄이기
batch_size = 16  # 또는 8

# 해결책 2: 샘플 수 줄이기
max_train_samples = 3000
```

### 2. Runtime Disconnected

```python
# Colab은 90분 idle시 연결 해제
# 해결: 코드 셀에 주기적 출력 추가

import time
for epoch in range(num_epochs):
    print(f"⏰ {time.strftime('%H:%M:%S')} - Epoch {epoch+1} 시작")
    # ... training ...
```

### 3. 너무 느림

```python
# 1. 샘플 수 대폭 줄이기
max_train_samples = 1000

# 2. Epoch 줄이기
num_epochs = 1

# 3. Pruning ratio 1개만
pruning_ratios = [0.3]

# 4. Log interval 늘리기
log_interval = 500
```

### 4. 패키지 설치 오류

```python
# 캐시 무시하고 재설치
!pip install --no-cache-dir transformers datasets torch
```

## Google Drive 연동 💾

실험 결과를 Google Drive에 자동 저장:

```python
# Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 결과 저장 경로 설정
output_dir = '/content/drive/MyDrive/dsp_experiments'
os.makedirs(output_dir, exist_ok=True)

# 결과 저장
output_file = os.path.join(output_dir, 'sst2_results.json')
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"✅ Drive에 저장됨: {output_file}")
```

## 체크리스트 ✅

실행 전 확인사항:

- [ ] GPU 런타임 활성화됨
- [ ] 패키지 설치 완료
- [ ] 메모리 여유 확인 (≥12GB)
- [ ] 실행 시간 예상 (≥30분)
- [ ] 결과 저장 경로 확인

## 팁 💡

1. **무료 GPU 시간 절약**
   - 개발/디버깅은 CPU로
   - 최종 실험만 GPU 사용

2. **중간 결과 저장**
   ```python
   # Epoch마다 결과 저장
   torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pt')
   ```

3. **W&B 로깅** (선택)
   ```python
   !pip install wandb
   import wandb

   wandb.init(project="dsp-experiments")
   wandb.log({"accuracy": acc, "step": step})
   ```

## 참고 자료 📚

- [Colab 공식 가이드](https://colab.research.google.com/notebooks/intro.ipynb)
- [Transformers 문서](https://huggingface.co/docs/transformers)
- [DSP 논문](https://arxiv.org/abs/2108.04657)

## 질문하기 ❓

문제가 발생하면:

1. **에러 메시지 전체 복사**
2. **실행 환경 정보 확인**
   ```python
   print(f"Torch: {torch.__version__}")
   print(f"Transformers: {transformers.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```
3. **Issue 생성** (GitHub repository)

---

Happy Experimenting! 🚀
