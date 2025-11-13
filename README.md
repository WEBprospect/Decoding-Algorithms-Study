# 디코딩 알고리즘 정리

이 문서는 언어 모델의 디코딩 알고리즘들을 정리한 문서입니다.

---

## 1. Greedy Search (그리디 서치)

### 개요
연속적인 모델 출력에서 이산적인 토큰을 얻는 가장 간단한 방법으로, 각 스텝마다 확률이 가장 높은 토큰을 탐욕적으로 선택하는 방법입니다.

### 특징
- **장점**: 구현이 간단하고 빠름
- **단점**: 매 단계에서 하나의 경로만 탐색하므로, 한 번 잘못된 토큰이 선택되면 이후 전체 문장이 비자연스럽게 이어질 수 있음

### 작동 방식
1. 각 타임스텝에서 프롬프트의 마지막 토큰에 대한 로짓을 선택
2. 소프트맥스를 적용해 확률 분포를 얻음
3. 확률이 가장 높은 토큰을 다음 토큰으로 선택
4. 입력 시퀀스에 추가한 후 이 과정을 반복

### 코드 예시

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

input_txt = "Transformers are the"
input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)

# 그리디 서치로 생성
output_greedy = model.generate(
    input_ids, 
    max_length=128, 
    do_sample=False  # 샘플링 비활성화 = 그리디 서치
)

print(tokenizer.decode(output_greedy[0]))
```

### 주요 파라미터
- `do_sample=False`: 샘플링을 비활성화하여 항상 가장 높은 확률의 토큰을 선택

---

## 2. Beam Search (빔 서치)

### 개요
그리디 서치의 한계를 보완한 알고리즘으로, 매 스텝마다 상위 **b개(빔 크기, beam size)**의 후보 토큰을 동시에 추적합니다.

### 특징
- **장점**: 여러 개의 가능한 시퀀스를 병렬로 확장·평가할 수 있어 탐색 공간이 넓어지고, 더 높은 확률을 갖는 문장을 찾을 가능성이 커짐
- **단점**: 빔 크기가 클수록 생성 과정이 느려짐

### 작동 방식
1. 매 스텝마다 상위 b개의 후보 토큰을 유지
2. 각 후보에 대해 다음 토큰의 확률을 계산
3. 전체 시퀀스의 로그 확률을 계산하여 최적 경로 선택
4. 최종적으로 가장 높은 로그 확률을 가진 시퀀스 반환

### 코드 예시

```python
# 빔 서치로 생성
output_beam = model.generate(
    input_ids,
    max_length=128,
    num_beams=5,      # 빔 크기 설정
    do_sample=False
)

# 로그 확률 계산 함수
def log_probs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logp_label

def sequence_logprob(model, labels, input_len=0):
    with torch.no_grad():
        output = model(labels)
        log_probs = log_probs_from_logits(output.logits[:,:-1,:], labels[:,1:])
        seq_log_prob = torch.sum(log_probs[:,input_len:])
    return seq_log_prob.cpu().numpy()

logp = sequence_logprob(model, output_beam, input_len=len(input_ids[0]))
print(f"로그 확률: {logp:.2f}")
```

### 반복 문제 해결
빔 서치도 텍스트가 반복되는 문제가 있을 수 있습니다. 이를 해결하기 위해 `no_repeat_ngram_size` 매개변수를 사용할 수 있습니다.

```python
output_beam = model.generate(
    input_ids,
    max_length=128,
    num_beams=5,
    do_sample=False,
    no_repeat_ngram_size=2  # 2-gram 반복 방지
)
```

### 주요 파라미터
- `num_beams`: 빔 크기 (기본값: 1, 그리디 서치와 동일)
- `no_repeat_ngram_size`: n-gram 반복 방지

---

## 3. Sampling-based Decoding (샘플링 기반 디코딩)

### 개요
확률 분포 전체에서 가장 높은 단어만 고르는 대신, 확률에 따라 랜덤하게 선택하는 방법입니다.

### 특징
- **장점**: 매번 다른 문장을 만들 수 있고, 창의적인 문장과 다양성이 생김
- **단점**: 확률이 너무 퍼져 있으면 엉뚱한 문장도 생성될 수 있음

---

### 3.1 Temperature Sampling (온도 샘플링)

#### 개요
Temperature는 확률 분포를 얼마나 "날카롭게" 또는 "퍼지게" 만들지 조절하는 파라미터입니다.

#### Temperature 값에 따른 효과

| Temperature | 효과 | 특징 |
|------------|------|------|
| T = 0.7 | 확률이 높은 단어 쪽으로 집중됨 | 보수적, 논리적 문장 |
| T = 1.0 | 기본값 (적당히 다양함) | 자연스러운 문장 |
| T = 1.5 | 확률이 퍼짐 | 창의적이지만 위험 |
| T → 0 | 사실상 그리디 서치 | 예측은 안정적, 다양성 없음 |

#### 코드 예시

```python
output_temp = model.generate(
    input_ids,
    max_length=max_length,
    do_sample=True,
    temperature=2.0,
    top_k=0
)
print(tokenizer.decode(output_temp[0]))
```

---

### 3.2 Top-k Sampling

#### 개요
확률이 가장 높은 k개 토큰에서만 샘플링하여 확률이 낮은 토큰을 피하는 방법입니다.

#### 특징
- 확률 분포의 롱테일을 잘라내고 확률이 가장 높은 토큰에서만 샘플링
- **문제점**: k값을 수동으로 지정해야 하며, 문장 생성 중 매 타임스텝마다 확률 분포가 달라도 항상 똑같은 k값을 적용해야 함

#### 코드 예시

```python
output_temp = model.generate(
    input_ids,
    max_length=max_length,
    do_sample=True,
    temperature=0.5,
    top_k=50  # 상위 50개 토큰에서만 샘플링
)
```

---

### 3.3 Top-p Sampling (Nucleus Sampling, 뉴클리어스 샘플링)

#### 개요
고정된 컷오프 값(k)을 선택하지 않고, 선택한 특정 확률 질량에 도달할 때까지 토큰을 선택하는 방법입니다.

#### 작동 방식
1. 확률에 따라 내림차순으로 모든 토큰을 정렬
2. 선택한 토큰의 확률 값이 p%에 도달할 때까지 리스트의 맨 위부터 토큰을 하나씩 추가
3. p값은 누적 확률 그래프의 수평선에 해당하며, 이 수평선 아래에 있는 토큰에서만 샘플링

#### 예시
```
토큰    확률    누적 확률
pizza   0.45    0.45
sushi   0.25    0.70
apple   0.15    0.85
banana  0.10    0.95  <- 여기서 stop! (p=0.90인 경우)
```

#### 특징
- 출력 분포에 따라 하나의 토큰이 될 수도 있고 백 개의 토큰이 될 수도 있음
- 확률이 높은 단어가 몇 개밖에 없다면 후보는 적음
- 확률이 고르게 퍼져 있다면 후보는 많음

#### 코드 예시

```python
output_topp = model.generate(
    input_ids,
    max_length=max_length,
    do_sample=True,
    top_p=0.90  # 누적 확률 90%까지의 토큰에서 샘플링
)
print(tokenizer.decode(output_topp[0]))
```

---

## 알고리즘 비교 요약

| 알고리즘 | 다양성 | 안정성 | 속도 | 사용 사례 |
|---------|--------|--------|------|----------|
| Greedy Search | 낮음 | 높음 | 빠름 | 번역, 요약 등 정확성이 중요한 경우 |
| Beam Search | 중간 | 높음 | 느림 | 번역, 요약 등 최적 해를 찾아야 하는 경우 |
| Temperature Sampling | 높음 | 중간 | 빠름 | 창의적 텍스트 생성 |
| Top-k Sampling | 중간 | 중간 | 빠름 | 균형잡힌 텍스트 생성 |
| Top-p Sampling | 높음 | 중간 | 빠름 | 자연스러운 대화 생성 |

---

## 참고사항

- 모든 알고리즘은 `transformers` 라이브러리의 `generate()` 함수를 통해 사용 가능
- 여러 파라미터를 조합하여 사용할 수 있음 (예: `temperature` + `top_p`)
- 실제 사용 시에는 작업의 특성에 맞는 알고리즘과 파라미터를 선택해야 함

