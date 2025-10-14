# Exercise Data Analysis Script

## 개요 (Overview)

이 MATLAB 스크립트는 운동 데이터를 자동으로 분석하고 PowerPoint 보고서를 생성합니다.

This MATLAB script automatically analyzes exercise data and generates a PowerPoint report.

## 기능 (Features)

1. **자동 파일 검색**: 모든 하위 폴더에서 "AAALogging_*.csv" 파일을 자동으로 찾습니다
2. **운동 강도 분석**: 다양한 운동 강도 지표를 계산합니다
3. **스펙트로그램 생성**: 가속도 데이터의 주파수 분석을 수행합니다
4. **자동 PPT 생성**: 분석 결과를 PowerPoint 프레젠테이션으로 자동 생성합니다

## 요구사항 (Requirements)

- MATLAB R2016b 이상
- Signal Processing Toolbox (스펙트로그램용)
- Statistics and Machine Learning Toolbox (선택사항)

## 사용 방법 (Usage)

1. MATLAB에서 스크립트가 있는 폴더로 이동
2. 명령 창에서 다음을 실행:
   ```matlab
   analyze_exercise_data
   ```

## 분석 지표 (Analysis Metrics)

스크립트는 다음과 같은 운동 강도 지표를 계산합니다:

### 1. 기본 통계량 (Basic Statistics)
- 평균 가속도 크기 (Mean Magnitude)
- 표준편차 (Standard Deviation)
- 최대/최소값 (Max/Min)

### 2. 에너지 관련 지표 (Energy Metrics)
- RMS (Root Mean Square) - 각 축별 및 전체
- SMA (Signal Magnitude Area)
- 총 에너지 (Total Energy)

### 3. 변동성 지표 (Variability Metrics)
- 분산 (Variance)
- IQR (Interquartile Range)
- Movement Variability Index
- Peak-to-Peak 진폭

### 4. 빈도 관련 지표 (Frequency Metrics)
- Zero Crossing Rate (ZCR)
- 스펙트로그램 분석

### 5. 통계적 특성 (Statistical Properties)
- 왜도 (Skewness)
- 첨도 (Kurtosis)
- 축 간 상관계수 (Correlation between axes)

### 6. 강도 분류 (Intensity Classification)
- Low: RMS < 0.5
- Moderate: 0.5 ≤ RMS < 1.5
- High: RMS ≥ 1.5

## 출력 파일 (Output Files)

스크립트 실행 후 다음 파일들이 생성됩니다:

1. **exercise_analysis_report.pptx**: 전체 분석 보고서
   - 첫 페이지: 전체 요약 (Summary)
   - 이후 페이지: 각 CSV 파일별 상세 분석 결과

2. **spectrogram_*.png**: 각 파일별 스펙트로그램 이미지
   - 3축 가속도 (Acc_X, Acc_Y, Acc_Z) 스펙트로그램
   - 가속도 크기 (Magnitude) 스펙트로그램

## CSV 파일 형식 (CSV File Format)

입력 CSV 파일은 다음 컬럼을 포함해야 합니다:
- `Acc_X`: X축 가속도 데이터
- `Acc_Y`: Y축 가속도 데이터
- `Acc_Z`: Z축 가속도 데이터

## 예제 (Example)

```matlab
% MATLAB 명령 창에서 실행
cd /path/to/your/data/directory
analyze_exercise_data

% 완료 후 다음 메시지가 표시됩니다:
% === Analysis Complete ===
% PowerPoint file saved as: exercise_analysis_report.pptx
```

## 문제 해결 (Troubleshooting)

### "No AAALogging_*.csv files found" 오류
- 현재 디렉토리와 하위 폴더에 CSV 파일이 있는지 확인하세요
- 파일명이 "AAALogging_"로 시작하고 ".csv"로 끝나는지 확인하세요

### "Missing required columns" 경고
- CSV 파일에 Acc_X, Acc_Y, Acc_Z 컬럼이 있는지 확인하세요
- 컬럼명의 대소문자가 정확한지 확인하세요

### PowerPoint 생성 오류
- MATLAB Report Generator가 설치되어 있는지 확인하세요
- `ver` 명령으로 설치된 툴박스를 확인할 수 있습니다

## 커스터마이징 (Customization)

### 샘플링 주파수 변경
스크립트의 `generate_spectrograms` 함수에서 `fs` 값을 수정하세요:
```matlab
fs = 100; % 기본값: 100 Hz, 필요에 따라 변경
```

### 강도 분류 기준 변경
`calculate_intensity_metrics` 함수의 분류 기준을 수정하세요:
```matlab
if metrics.rms_total < 0.5
    metrics.intensity_level = 'Low';
elseif metrics.rms_total < 1.5
    metrics.intensity_level = 'Moderate';
else
    metrics.intensity_level = 'High';
end
```

## 라이선스 (License)

이 스크립트는 교육 및 연구 목적으로 자유롭게 사용할 수 있습니다.
