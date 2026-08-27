# test/ — 셀피 한 장 없이 86개

**테스트는 이미지를 요구하지 않는다.** 합성 얼굴 픽스처로 전 구간을 검증한다.
이것이 `core/` 를 numpy 전용으로 묶은 두 번째 이유다(첫 번째는 C++ 포팅).

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pytest test/test_contract.py -v
.venv/Scripts/python.exe -m pytest test/test_measure.py::test_no_lesions_on_clean_skin -v
.venv/Scripts/python.exe -m pytest -q -k melanin
```

| 파일 | 개수 | 다루는 것 |
|---|---|---|
| `test_contract.py` | 11 | ★ 계층 규칙을 AST 로 강제 |
| `test_color.py` | 18 | 불변성 property · 색소 방향 벡터 |
| `test_mask.py` | 16 | 폴리곤 래스터화 · 색도/정반사 게이트 · 부위 배타성 |
| `test_measure.py` | 23 | 2단계 기준선 · DoG 복원 · null 전파 |
| `test_quality.py` | 18 | 게이트 우선순위 · ReasonCode |
| `conftest.py` | — | 합성 478 랜드마크 · 물리적으로 올바른 주입 |

## `test_contract.py` 는 문서가 아니라 강제다

`docs/CONTRACT.md` 의 1·5·9번을 실제로 막는다 — 문서에만 적어두면 반드시 새어 나간다.

- `core/` 가 numpy·stdlib 외를 import 하면 실패
- `web/` 이 `core/` 를 직접 import 하면 실패
- `config/default.toml` 과 `core/config.py` dataclass 기본값이 어긋나면 실패
- 오타 난 설정 키가 `ValueError` 를 안 내면 실패
- 외부 데이터 파일이 흡광계수 CSV 하나가 아니면 실패

## 합성 주입은 반드시 물리적으로 옳아야 한다

> **붉은 원을 알파 블렌딩하지 말 것.** 비물리적이고 **검출기를 실제보다 좋아 보이게 만든다.**

로그 밀도 공간에서 색소 방향으로 더한다:

```python
def inject(linear, amount, direction):
    """D' = D + amount * v   ->   R' = R * 10^(-amount * v)"""
    factor = np.power(10.0, -amount[..., None] * direction[None, None, :])
    return np.clip(linear * factor, 1e-4, 1.0).astype(np.float32)
```

방향 벡터는 `v_hemoglobin` / `v_melanin` 픽스처에서 오고, 그건 흡광계수 CSV +
**진피 광로장 가중**에서 계산된다. 가중을 빼면 `v_h` 가 물리적으로 뒤집힌다.

## 픽스처

| 픽스처 | 내용 |
|---|---|
| `landmarks` | 합성 478 랜드마크 (정면·정규 프레임 좌표) |
| `clean_face` | 병변 없는 균일 피부 — 오검출 상한 측정용 |
| `frame_valid` | 유효성 마스크 |
| `v_hemoglobin` · `v_melanin` | 두 색소의 log-RGB 방향 |

## 테스트를 쓸 때 주의할 것

**기대값이 아니라 결과를 검증한다.** 특정 메커니즘이 발동하기를 요구하면 상류에서 이미
처리된 경우 테스트가 거짓 실패한다 — 실제로 겪었다. 멜라닌 블롭은 ERI 회귀가 이미 지워버려
병변 cascade 가 발동조차 하지 않았다. **결과("검출되지 않는다")를 보는 테스트 하나와,
규칙 자체를 보는 단위 테스트 하나로 나눈다.**

**포화는 물리다.** 밝기 불변성 테스트에서 값이 1.0 에 클리핑되면 비율이 깨지는데,
그건 버그가 아니라 클리핑 게이트가 존재하는 이유다. 픽스처 범위를 0.03~0.30 으로 좁히고,
포화가 불변성을 깨뜨린다는 사실 자체를 별도 테스트로 남긴다.

## 실사진이 없다는 것

여기서 통과하는 86개는 **합성 데이터 기준이고 실사보다 쉽다.**
`docs/LIMITS.md` §2 의 실측 칸은 여전히 대부분 _미측정_ 이다.
**테스트 통과를 성능 근거로 인용하지 않는다.**
