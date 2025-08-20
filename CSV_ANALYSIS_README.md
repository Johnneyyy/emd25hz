# CSV 신호 분석 가이드

## 개요
CSV 파일에서 신호를 로드하여 STFT(Short-Time Fourier Transform)와 필터뱅크(Filter Bank) 방식으로 분석하는 MATLAB 스크립트입니다.

## 파일 구성

### 1. `csv_signal_analysis.m` (메인 분석 스크립트)
- CSV 파일에서 신호를 로드
- STFT와 필터뱅크 분석 수행
- 결과 시각화 및 저장

### 2. `generate_sample_csv.m` (샘플 데이터 생성)
- 다양한 테스트 신호 생성
- CSV 파일로 저장

## 사용 방법

### 1단계: 샘플 CSV 파일 생성 (선택사항)
```matlab
% MATLAB에서 실행
generate_sample_csv
```

생성되는 파일들:
- `signal_stationary.csv`: 정상 신호 (고정 주파수)
- `signal_chirp.csv`: 주파수 변조 신호
- `signal_timevarying.csv`: 시변 신호
- `signal_pulse.csv`: 펄스 신호
- `signal_am_modulated.csv`: AM 변조 신호
- `signal_biosignal.csv`: 생체 신호 유사
- `signal_multichannel.csv`: 다중 채널 신호

### 2단계: 신호 분석 실행
```matlab
% MATLAB에서 실행
csv_signal_analysis
```

프로그램 실행 시:
1. 기본적으로 `signal_data.csv` 파일을 찾음
2. 파일이 없으면 샘플 생성 옵션 제공
3. 분석 완료 후 결과 저장 옵션 제공

### 3단계: 커스텀 CSV 파일 사용
자신의 CSV 파일을 사용하려면:

1. CSV 파일 형식:
   - 2열 형식: [시간, 신호]
   - 1열 형식: [신호] (샘플링 주파수 입력 필요)

2. 스크립트 수정:
```matlab
% csv_signal_analysis.m 파일의 8번째 줄 수정
csv_file = 'your_file_name.csv';  % 자신의 파일명으로 변경
```

## 분석 결과

### 시각화 내용
1. **원본 신호**: 시간 도메인 신호
2. **FFT 스펙트럼**: 전체 주파수 스펙트럼
3. **신호 통계**: 평균, 표준편차, 최대/최소값, RMS
4. **STFT 스펙트로그램**: 시간-주파수 분석
5. **필터뱅크 스펙트로그램**: 실시간 주파수 분석
6. **시간 해상도 비교**: 특정 주파수 대역의 시간 변화
7. **주파수 해상도 비교**: 특정 시간의 주파수 스펙트럼
8. **위상 정보**: 필터뱅크 위상 분석
9. **성능 메트릭**: 두 방법의 성능 비교

### 저장되는 파일
분석 결과 저장 시 `analysis_results/` 폴더에:
- `*_stft_result.mat`: STFT 분석 결과
- `*_filterbank_result.mat`: 필터뱅크 분석 결과
- `*_analysis_plot.png`: 분석 그래프

## 주요 파라미터 조정

### STFT 파라미터
```matlab
window_length = round(fs * 2);  % 윈도우 크기 (초 단위)
overlap = round(window_length * 0.9);  % 오버랩 비율
```

### 필터뱅크 파라미터
```matlab
filter_order = 4;  % 필터 차수
n_bands = 20;  % 주파수 대역 수
```

## STFT vs 필터뱅크 비교

### STFT (Short-Time Fourier Transform)
**장점:**
- 높은 주파수 해상도 (긴 윈도우 사용 시)
- FFT 기반으로 계산 효율적
- 표준화된 분석 방법

**단점:**
- 고정된 윈도우 크기 (시간-주파수 해상도 트레이드오프)
- 실시간 처리에 제약

**최적 사용 조건:**
- 오프라인 분석
- 정상 신호
- 높은 주파수 해상도가 필요한 경우

### 필터뱅크 (Filter Bank)
**장점:**
- 우수한 시간 해상도
- 실시간 처리 가능
- 각 대역 독립적 처리

**단점:**
- 필터 설계에 따른 성능 변화
- 높은 계산 복잡도 (많은 대역 사용 시)

**최적 사용 조건:**
- 실시간 모니터링
- 시변 신호
- 특정 주파수 대역 추적

## 문제 해결

### 오류: "Index exceeds matrix dimensions"
- CSV 파일 형식 확인 (최소 1열 이상)
- 샘플링 주파수가 신호에 적합한지 확인

### 오류: "Invalid filter specifications"
- 주파수 범위가 Nyquist 주파수 내에 있는지 확인
- 필터 차수를 낮춰보기 (2 또는 4 권장)

### 메모리 부족
- 신호 길이 줄이기
- 다운샘플링 적용
- 주파수 대역 수 감소

## 예제 실행

### 기본 예제
```matlab
% 1. 샘플 생성
generate_sample_csv

% 2. 정상 신호 분석
csv_file = 'signal_stationary.csv';
csv_signal_analysis

% 3. 시변 신호 분석
csv_file = 'signal_timevarying.csv';
csv_signal_analysis
```

### 고급 예제 (파라미터 조정)
```matlab
% csv_signal_analysis.m 수정
% 라인 47-50: 주파수 범위 조정
freq_start = 0.1;
freq_end = 10;  % 더 넓은 주파수 범위
n_bands = 30;   % 더 많은 주파수 대역

% 라인 58: 윈도우 크기 조정
window_length = round(fs * 5);  % 5초 윈도우
```

## 참고 자료
- STFT: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
- Filter Bank: https://en.wikipedia.org/wiki/Filter_bank
- Hilbert Transform: https://en.wikipedia.org/wiki/Hilbert_transform