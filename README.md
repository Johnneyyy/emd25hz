# EMD/EEMD/FIF 분석 도구

0.2Hz~4Hz 주파수 대역을 대상으로 하는 EMD(Empirical Mode Decomposition), EEMD(Ensemble EMD), FIF(Fast Iterative Filtering) 분석 도구입니다.

## 주요 특징

- **CSV 파일 입력**: 다양한 CSV 파일에서 원하는 컬럼을 선택하여 분석
- **주파수 대역 특화**: 0.2Hz~4Hz 주파수 대역에 최적화된 분석
- **다중 분해 방법**: EMD, EEMD, FIF 세 가지 방법 지원
- **Hilbert 변환 분석**: 순간 주파수 및 에너지 분석
- **1초 단위 시간축**: 실시간 데이터 분석에 적합한 시간 해상도
- **직관적인 GUI**: 사용자 친화적인 그래픽 인터페이스
- **상세한 시각화**: 분석 결과의 종합적인 시각화
- **결과 저장**: 분석 결과를 CSV 파일로 저장

## 빠른 시작

### 1. 패키지 설치
```bash
sudo apt update
sudo apt install -y python3-numpy python3-pandas python3-matplotlib python3-scipy python3-tk
```

### 2. 샘플 데이터 생성
```bash
python3 simple_sample_generator.py
```

### 3. 분석 도구 실행
```bash
python3 emd_analysis_tool.py
```

## 파일 구조

```
workspace/
├── emd_analysis_tool.py          # 메인 분석 도구 (GUI)
├── simple_sample_generator.py    # 간단한 샘플 데이터 생성기
├── generate_sample_data.py       # 상세한 샘플 데이터 생성기
├── sample_emd_data.csv           # 생성된 샘플 데이터
├── requirements.txt              # Python 패키지 요구사항
├── README_EMD_Analysis.md        # 상세한 사용 설명서
└── README.md                     # 이 파일
```

## 기본 사용법

1. **분석 도구 실행**: `python3 emd_analysis_tool.py`
2. **CSV 파일 선택**: GUI에서 "CSV 파일 선택" 버튼 클릭
3. **분석 설정**: 
   - 분석할 컬럼 선택
   - 샘플링 주파수 설정 (기본: 1.0Hz)
   - 주파수 대역 설정 (기본: 0.2Hz~4.0Hz)
4. **분석 방법 선택**: EMD, EEMD, FIF 중 원하는 방법 체크
5. **분석 실행**: "분석 실행" 버튼 클릭
6. **결과 확인**: 자동으로 표시되는 시각화 결과 확인
7. **결과 저장**: "결과 저장" 버튼으로 CSV 파일로 저장

## 분석 결과

### IMF (Intrinsic Mode Functions)
- **IMF 1**: 가장 높은 주파수 성분
- **IMF 2, 3, ...**: 점진적으로 낮은 주파수 성분들
- **Residual**: 잔여 성분 (트렌드)

### Hilbert 분석
- **순간 주파수**: 시간에 따른 주파수 변화
- **순간 진폭**: 시간에 따른 진폭 변화
- **에너지**: 각 IMF의 총 에너지

### 시각화
- **종합 비교**: 세 방법의 IMF, 에너지 분포, Hilbert 스펙트럼 비교
- **상세 분석**: 각 방법별 IMF와 잔여 성분의 상세 파형

## 적용 분야

### 생체신호 분석
- 심박수 변동성 (HRV) 분석
- 호흡 신호 분석
- 뇌파 (EEG) 분석

### 센서 데이터 분석
- 진동 신호 분석
- 환경 모니터링 데이터
- IoT 센서 데이터

### 시계열 데이터 분석
- 경제 데이터 분석
- 기후 데이터 분석
- 트렌드 분석

## 기술적 세부사항

### EMD (Empirical Mode Decomposition)
- 비선형, 비정상 신호에 적합
- Sifting 과정을 통한 IMF 추출
- 모드 혼합 문제 가능성

### EEMD (Ensemble EMD)
- 백색 잡음 추가로 모드 혼합 문제 완화
- 앙상블 평균으로 안정적인 결과
- 계산 시간이 상대적으로 오래 걸림

### FIF (Fast Iterative Filtering)
- 빠른 계산 속도
- 안정적인 분해 결과
- 마스크 기반 필터링

## 샘플 데이터

생성된 샘플 데이터에는 다음이 포함됩니다:

1. **HRV_Signal**: 심박수 변동성 유사 신호 (0.3Hz, 0.8Hz, 1.5Hz 성분)
2. **Respiratory_Signal**: 호흡 관련 신호 (0.25Hz, 0.4Hz 성분)
3. **Complex_Biosignal**: 복합 생체 신호 (0.3Hz, 1.0Hz, 2.2Hz, 3.7Hz 성분)

## 문제 해결

### 일반적인 오류
- **파일 로드 오류**: CSV 파일 형식과 경로 확인
- **분석 오류**: 숫자형 데이터와 샘플링 주파수 확인
- **메모리 오류**: 데이터 크기 줄이거나 EEMD 앙상블 수 조정

### 성능 최적화
- 데이터 크기: 1000~10000 샘플 권장
- 주파수 설정: 분석 대상의 2배 이상 샘플링 주파수
- 방법 선택: 빠른 분석은 FIF, 정확한 분석은 EEMD

## 라이선스

MIT 라이선스

---

**참고**: 이 도구는 연구 및 교육 목적으로 개발되었습니다. 의료 진단이나 상업적 용도로 사용하기 전에 충분한 검증이 필요합니다.

더 자세한 정보는 `README_EMD_Analysis.md` 파일을 참조하세요.