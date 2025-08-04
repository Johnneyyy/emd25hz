# EMD 계열 방법을 이용한 25Hz 신호 IMF-Hilbert 에너지 분석

이 프로젝트는 25Hz 신호에 대해 EMD(Empirical Mode Decomposition) 계열 방법들을 사용하여 IMF(Intrinsic Mode Function) 분해 및 Hilbert 변환 기반 에너지 분석을 수행하는 MATLAB 코드 모음입니다.

## 📁 파일 구조

### 주요 분석 스크립트
- `emd_hilbert_analysis.m` - EMD와 EEMD를 이용한 기본 분석
- `fif_analysis.m` - FIF(Fast Iterative Filtering) 기반 분석
- `comprehensive_emd_comparison.m` - 모든 방법들의 종합 비교 분석

### 생성되는 결과 파일
- `emd_hilbert_results.mat` - EMD/EEMD 분석 결과
- `fif_hilbert_results.mat` - FIF 분석 결과
- `comprehensive_emd_comparison_results.mat` - 종합 비교 분석 결과

## 🚀 사용 방법

### 1. 기본 EMD/EEMD 분석
```matlab
run('emd_hilbert_analysis.m')
```

### 2. FIF 분석
```matlab
run('fif_analysis.m')
```

### 3. 종합 비교 분석 (권장)
```matlab
run('comprehensive_emd_comparison.m')
```

## 📊 분석 내용

### 신호 특성
- **주파수**: 25Hz 주성분 (진폭/주파수 변조)
- **추가 성분**: 5Hz, 50Hz, 75Hz 성분
- **비선형 특성**: 처프 신호, 진폭 변조
- **노이즈**: 백색 가우시안 노이즈
- **길이**: 10초 (샘플링 주파수: 1000Hz)

### 분석 방법

#### 1. EMD (Empirical Mode Decomposition)
- 신호를 여러 IMF로 분해
- 각 IMF는 단일 주파수 성분을 나타냄
- 비선형, 비정상 신호에 적합

#### 2. EEMD (Ensemble Empirical Mode Decomposition)
- EMD의 모드 혼합 문제 해결
- 여러 노이즈 추가 신호의 앙상블 평균
- 더 안정적인 분해 결과

#### 3. FIF (Fast Iterative Filtering)
- EMD의 개선된 버전
- 반복적 필터링 기반
- 계산 효율성과 안정성 향상

### Hilbert 변환 분석
- **순간 주파수**: 각 IMF의 시간에 따른 주파수 변화
- **순간 진폭**: 각 IMF의 시간에 따른 진폭 변화
- **에너지 분석**: 각 IMF의 에너지 분포
- **Hilbert 스펙트럼**: 시간-주파수-진폭 3차원 표현

## 📈 결과 시각화

### 1. 시간 도메인 분석
- 원본 신호 및 각 IMF 시각화
- 25Hz 성분 식별 및 분석

### 2. 주파수 도메인 분석
- 파워 스펙트럼 밀도
- 스펙트로그램
- Hilbert 스펙트럼

### 3. 에너지 분석
- IMF별 에너지 분포
- 25Hz 성분의 에너지 기여도
- 방법별 에너지 분해 비교

### 4. 성능 비교
- 계산 시간
- 재구성 오차
- 모드 혼합 지수 (Mode Mixing Index)
- 25Hz 성분 추출 정확도

## 🔧 시스템 요구사항

### MATLAB 버전
- MATLAB R2018b 이상 권장

### 필요한 툴박스
- **Signal Processing Toolbox** (권장)
  - `emd()` 함수 사용
  - `pwelch()`, `spectrogram()` 함수
- **기본 MATLAB** (최소 요구사항)
  - 사용자 정의 EMD 구현 포함

### 메모리 요구사항
- 최소 4GB RAM 권장
- EEMD 분석 시 더 많은 메모리 필요

## 📋 분석 결과 해석

### 출력 정보
```
=== 종합 분석 결과 요약 ===
방법        IMF수   계산시간(s)   재구성오차   MMI      25Hz위치   25Hz에너지
EMD         6       0.15          0.000001     0.0234   3          0.4521
EEMD        6       7.82          0.000002     0.0156   3          0.4487
FIF         5       0.23          0.000003     0.0198   2          0.4502
```

### 성능 지표 설명
- **IMF수**: 분해된 고유 모드 함수의 개수
- **계산시간**: 분해 수행 시간 (초)
- **재구성오차**: 원본 신호 재구성의 정규화된 오차
- **MMI**: 모드 혼합 지수 (낮을수록 좋음)
- **25Hz위치**: 25Hz 성분에 해당하는 IMF 번호
- **25Hz에너지**: 25Hz IMF의 에너지 값

## 🔍 고급 기능

### 사용자 정의 파라미터
```matlab
% EEMD 파라미터 조정
num_ensembles = 100;    % 앙상블 개수 (더 높을수록 정확하지만 느림)
noise_std = 0.1;        % 노이즈 표준편차

% FIF 파라미터 조정
options.delta = 0.001;          % 수렴 임계값
options.NIMFs = 8;              % 최대 IMF 개수
options.MaxInner = 200;         % 최대 내부 반복
```

### 신호 특성 변경
```matlab
% 다른 주파수 성분으로 변경
f1 = 30;    % 주요 주파수를 30Hz로 변경
f2 = 10;    % 저주파 성분을 10Hz로 변경

% 신호 길이 변경
t = 0:1/fs:5-1/fs;  % 5초로 단축
```

## 📚 참고문헌

1. Huang, N. E., et al. (1998). "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis."
2. Wu, Z., & Huang, N. E. (2009). "Ensemble empirical mode decomposition: a noise-assisted data analysis method."
3. Cicone, A., et al. (2016). "Adaptive local iterative filtering for signal decomposition and instantaneous frequency analysis."

## 🐛 문제 해결

### 일반적인 오류
1. **EMD 함수 없음**: Signal Processing Toolbox가 없는 경우 자동으로 사용자 정의 구현 사용
2. **메모리 부족**: EEMD 앙상블 개수를 줄이거나 신호 길이 단축
3. **수렴 문제**: FIF 파라미터 조정 (delta 값 증가, MaxInner 감소)

### 성능 최적화
- EEMD 앙상블 개수 조정으로 정확도와 속도 균형
- 병렬 처리 가능한 경우 `parfor` 사용 고려
- 메모리 사용량 모니터링

## 📞 지원

추가 질문이나 문제가 있는 경우:
1. MATLAB 도움말 문서 참조: `help emd`, `help hilbert`
2. 코드 내 주석 확인
3. 매개변수 조정을 통한 실험적 접근

---

**주의**: 이 코드는 교육 및 연구 목적으로 작성되었습니다. 실제 응용에서는 신호 특성에 맞는 파라미터 조정이 필요할 수 있습니다.