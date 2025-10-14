# 운동 데이터 분석 - 빠른 시작 가이드

## 📋 생성된 파일 목록

1. **analyze_exercise_data.m** - 메인 분석 스크립트
2. **create_sample_data.m** - 샘플 데이터 생성 스크립트
3. **run_example.m** - 전체 워크플로우 예제
4. **EXERCISE_ANALYSIS_README.md** - 상세 설명서

## 🚀 사용 방법

### 옵션 1: 예제 데이터로 테스트 (권장)

MATLAB에서 다음을 실행:
```matlab
run_example
```

이 명령은 자동으로:
- 샘플 운동 데이터 생성 (걷기, 조깅, 달리기)
- 데이터 분석 수행
- PowerPoint 보고서 생성

### 옵션 2: 실제 데이터 분석

1. 운동 데이터 CSV 파일을 작업 폴더에 배치
   - 파일명: `AAALogging_*.csv` 형식
   - 필수 컬럼: `Acc_X`, `Acc_Y`, `Acc_Z`

2. MATLAB에서 실행:
   ```matlab
   analyze_exercise_data
   ```

## 📊 출력 결과

분석 완료 후 다음 파일들이 생성됩니다:

1. **exercise_analysis_report.pptx**
   - 페이지 1: 전체 요약 (파일 개수, 평균 통계, 강도 분포)
   - 페이지 2~N: 각 파일별 상세 분석 및 스펙트로그램

2. **spectrogram_*.png**
   - 각 운동 데이터의 주파수 분석 이미지

## 📈 분석 지표

### 1. 강도 지표
- RMS (Root Mean Square)
- SMA (Signal Magnitude Area)
- 총 에너지
- Movement Variability Index

### 2. 통계 지표
- 평균, 표준편차, 최대/최소값
- 분산, IQR
- 왜도, 첨도

### 3. 주파수 분석
- 스펙트로그램 (시간-주파수 분석)
- Zero Crossing Rate

### 4. 강도 분류
- Low (낮음): RMS < 0.5
- Moderate (보통): 0.5 ≤ RMS < 1.5
- High (높음): RMS ≥ 1.5

## 💡 팁

### 샘플링 주파수 변경
`analyze_exercise_data.m` 파일의 226번째 줄:
```matlab
fs = 100; % 현재 센서의 샘플링 주파수로 변경
```

### 강도 분류 기준 조정
`analyze_exercise_data.m` 파일의 171-177번째 줄에서 기준값 수정

### CSV 파일 형식
```csv
timestamp,Acc_X,Acc_Y,Acc_Z
0,-0.123,0.456,9.81
1,-0.145,0.432,9.79
...
```

## ⚠️ 문제 해결

### "No files found" 오류
- 파일명이 `AAALogging_`로 시작하는지 확인
- 현재 폴더 또는 하위 폴더에 파일이 있는지 확인

### PowerPoint 생성 오류
- MATLAB Report Generator 툴박스 확인:
  ```matlab
  ver
  ```

### 메모리 부족
- 대용량 파일 처리 시 발생 가능
- 데이터를 여러 배치로 나누어 처리

## 📞 추가 정보

자세한 내용은 `EXERCISE_ANALYSIS_README.md` 파일을 참조하세요.

## 🎯 다음 단계

1. ✅ `run_example` 실행하여 동작 확인
2. ✅ 생성된 PPT 파일 확인
3. ✅ 실제 데이터로 분석 수행
4. ✅ 필요시 파라미터 조정 및 커스터마이징
