# 디코딩 알고리즘 스터디

#개요
이 문서는 언어 모델의 디코딩 알고리즘들을 정리한 문서입니다.

---

## 1. Greedy Search (그리디 서치)

### 개념
그리디 서치는 매 스텝에서 확률이 가장 높은 단어 하나만 선택해서 문장을 생성하는 알고리즘이다.
모델이 출력하는 전체 단어 분포(softmax 확률)중에서 
가장 큰 확률을 가진 토큰 하나를 선택하여 입력 뒤에 붙이고 그 다음 스텝도 같은 방식으로 반복한다.

---

### 식

## GPT는 토큰 단위로 다음 확률을 예측함
P(yt​∣y<t​,x)

-- y<t : 지금까지 생성된 토큰들

-- yt : 다음 토큰 

## Gready Search 선택 규칙

 yt​=argmax​P(w∣y<t​,x)

--각 스텝에서 확률이 가장 높은 단어 w를 선택한다.

## 전체 문장 확률을 생성하는 식 

1.문장을 T토큰까지 생성한다고 할떄 
  Y^={y^​1​,y^​2​,…,y^​T​}

2.각 토큰은 
  y^​t​=argmax​P(w∣y^​<t​,x)

3.전체 문장은 
  Y^=argw1​max​P(w1​)⋅argw2​max​P(w2​∣w1​)⋅argw3​max​P(w3​∣w1​,w2​)⋅…

---

## 문장 생성 예시 

문장 시작 
The cat

다음 토큰 확률 : 
| 토큰   | 확률   |
| ---- | ---- |
| sat  | 0.55 |
| is   | 0.30 |
| with | 0.08 |
| and  | 0.02 |

-- 그리디 서치 결과

   The cat sat

계속 동일방식으로 진행 ..
---

### 특징
- **장점**
- 1.구현이 간단하고 속도가 빠릅니다
- 2.그리디 서치 디코딩은 후보가 1개라서 메모리를 거의 차지를 않합니다
- 3.매 살행마다 항상 같은 문장이 나옵니다
       
- **단점**
- 1.그리디는 "현재 가장 좋은 선택"만 고르기 때문에
             전체 문장의 품질은 형편 없을수 있습니다
 
- 2.반복되는 문장이 나올수 있습니다
- 3.창의성 부족
    -->확률이 가장 높은것만 계속 고르기 때문에 다양하고 창의적인 문장은 거의 나오지 않습니다
- 4.초기 오류가 그 뒤의 모든 선택이 영향을 받아 전체 문장이 엉망이 될수 있음
---

### 작동 방식
1. 각 타임스텝에서 프롬프트의 마지막 토큰에 대한 로짓을 선택합니다.
2. 소프트맥스를 적용해 확률 분포를 얻습니다
3. 확률이 가장 높은 토큰을 다음 토큰으로 선택합니다
4. 입력 시퀀스에 추가한 후 이 과정을 반복합니다
---

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
---
### 주요 파라미터
- `do_sample=False`: 샘플링을 비활성화하여 항상 가장 높은 확률의 토큰을 선택

---


## 2. Beam Search (빔 서치)

### 개념
빔 서치는 텍스트 생성 시 매 스텝에서 하나만 선택하는 그리디 서치와 달리,상쉬 b개의 후보 시퀀스(빔)를 유지하면서 가장 가능성 높은 문장을 탐색하는
디코딩 알고리즘 입니다.각 타임스텝마다 확률이 높은 상위 b개의 다음 토큰을 확장하여 새로운 후보 시퀀스를 만들고,그 전체 후보 중 상위 b개만 남겨 
다음 단계로 진행한다. 이렇게 함으로써 그리디 서치의 단점인 "초기 선택 실수"를 완화하고 더 전역적으로 좋은 문장을 찾을 가능성이 높다.


### GPT의 다음 토큰 확률

- 모델은 다음 확률을 예측한다.
- P(yt​∣y<t​,x)

### Beam Search 선택 규칙 

-각 후보 시퀀스의 점수를 다음과 같이 누적한다.
<img width="382" height="90" alt="image" src="https://github.com/user-attachments/assets/f3b88b6d-b410-4985-a058-f743f236a7e4" />


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

