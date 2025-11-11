# 스마트워치 가속도계 데이터로부터 이동 속도 계산

이 프로젝트는 스마트워치에서 25Hz로 샘플링된 가속도계 데이터(x, y, z축)로부터 사람의 이동 속도를 계산하는 MATLAB 함수를 제공합니다.

## 주요 기능

- **다양한 활동 패턴 지원**: 걷기, 달리기, 워치를 보면서 뛰기 등 다양한 상황에서 속도 계산
- **중력 성분 제거**: 고역 통과 필터 또는 추세 제거를 통한 중력 성분 제거
- **드리프트 보정**: 적분 과정에서 발생하는 드리프트를 자동으로 보정
- **활동 분류**: 가속도 데이터를 기반으로 정지/걷기/달리기 자동 분류
- **상세한 결과 분석**: 속도, 가속도, 활동 분류 등 다양한 정보 제공

## 파일 구조

- `calculate_movement_speed.m` - 메인 함수: 가속도계 데이터로부터 속도 계산
- `generate_sample_accelerometer_data.m` - 샘플 데이터 생성 함수 (테스트용)
- `test_speed_calculation.m` - 테스트 스크립트: 다양한 시나리오 테스트

## 사용 방법

### 기본 사용법

```matlab
% 가속도계 데이터 준비 (x, y, z축, 단위: m/s²)
acc_x = ...; % x축 가속도 데이터
acc_y = ...; % y축 가속도 데이터
acc_z = ...; % z축 가속도 데이터
fs = 25;     % 샘플링 주파수 (Hz)

% 속도 계산
[speed, velocity, filtered_acc, results] = calculate_movement_speed(acc_x, acc_y, acc_z, fs);

% 결과 확인
fprintf('평균 속도: %.2f m/s (%.2f km/h)\n', mean(speed), mean(speed)*3.6);
```

### 고급 사용법 (파라미터 조정)

```matlab
[speed, velocity, filtered_acc, results] = calculate_movement_speed(...
    acc_x, acc_y, acc_z, fs, ...
    'gravity_removal', 'highpass', ...      % 중력 제거 방법: 'highpass', 'detrend', 'none'
    'cutoff_freq', 0.1, ...                 % 고역 통과 필터 차단 주파수 (Hz)
    'drift_correction', 'highpass', ...     % 드리프트 보정 방법: 'highpass', 'detrend', 'none'
    'integration_window', 5, ...           % 적분 윈도우 길이 (초)
    'min_speed_threshold', 0.1, ...         % 최소 속도 임계값 (m/s)
    'calibration_time', 2);                 % 캘리브레이션 시간 (초)
```

### g 단위 데이터 사용

```matlab
% 가속도계 데이터가 g 단위인 경우
[speed, velocity, filtered_acc, results] = calculate_movement_speed(...
    acc_x, acc_y, acc_z, fs, ...
    'unit', 'g');  % 단위를 'g'로 지정
```

## 함수 출력

### 주요 출력값

- `speed`: 이동 속도 (m/s) - 시간에 따른 속도 벡터
- `velocity`: 속도 벡터 (m/s) - [vx, vy, vz] 형태의 N×3 행렬
- `filtered_acc`: 필터링된 가속도 데이터 (m/s²) - [ax, ay, az] 형태의 N×3 행렬
- `results`: 상세 결과 구조체

### results 구조체 내용

```matlab
results.time              % 시간 벡터 (초)
results.acceleration      % 가속도 관련 데이터
results.velocity          % 속도 관련 데이터
results.speed             % 속도 (m/s)
results.activity_type     % 활동 분류 (0: 정지, 1: 걷기, 2: 달리기)
results.movement_energy   % 움직임 에너지
results.parameters        % 사용된 파라미터
results.statistics        % 통계 정보
```

## 테스트 실행

```matlab
% 테스트 스크립트 실행
run('test_speed_calculation.m')
```

이 스크립트는 다음 시나리오를 테스트합니다:
1. 걷기
2. 달리기
3. 혼합 (걷기 + 달리기)
4. 워치를 보면서 뛰기

각 시나리오에 대해:
- 원본 및 필터링된 가속도 데이터 시각화
- 계산된 속도 vs 실제 속도 비교
- 활동 분류 결과
- 통계 정보 출력

## 알고리즘 설명

### 1. 중력 성분 제거
- 고역 통과 필터를 사용하여 중력 성분 제거
- 또는 선형 추세 제거 방법 사용 가능

### 2. 노이즈 필터링
- 이동 평균 필터를 사용하여 노이즈 제거

### 3. 가속도 적분
- 슬라이딩 윈도우를 사용한 적분
- 사다리꼴 법칙을 사용한 수치 적분

### 4. 드리프트 보정
- 적분 과정에서 발생하는 저주파 드리프트 제거
- 고역 통과 필터 또는 추세 제거 사용

### 5. 속도 계산
- 속도 벡터의 크기 계산
- 최소 속도 임계값 적용
- 이동 평균으로 부드럽게 만들기

## 파라미터 조정 가이드

### 샘플링 주파수가 낮은 경우 (25Hz)
- `integration_window`: 5-10초로 증가 권장
- `cutoff_freq`: 0.1-0.2 Hz로 설정
- `drift_correction`: 'highpass' 사용 권장

### 노이즈가 많은 경우
- `filter_type`: 'butter' 사용
- `min_speed_threshold`: 0.2-0.3 m/s로 증가

### 빠른 움직임 (달리기) 위주인 경우
- `cutoff_freq`: 0.2-0.3 Hz로 증가
- `integration_window`: 3-5초로 감소

### 느린 움직임 (걷기) 위주인 경우
- `cutoff_freq`: 0.05-0.1 Hz로 감소
- `integration_window`: 5-10초로 증가

## 주의사항

1. **정확도 제한**: 가속도계만으로는 절대 속도를 정확히 계산하기 어렵습니다. 이 함수는 상대적인 속도 변화를 추정합니다.

2. **초기 조건**: 적분 과정에서 초기 속도가 0이라고 가정합니다. 실제로는 초기 속도를 알 수 없으므로 절대 속도보다는 속도 변화에 초점을 맞춥니다.

3. **드리프트**: 적분 과정에서 드리프트가 발생할 수 있으므로, `drift_correction` 파라미터를 적절히 설정하는 것이 중요합니다.

4. **워치 위치**: 워치가 손목에 착용된 경우, 손목의 회전 운동이 속도 계산에 영향을 줄 수 있습니다. 이 함수는 이를 최소화하기 위한 처리를 포함하고 있습니다.

## 시스템 요구사항

- MATLAB R2018b 이상
- Signal Processing Toolbox (권장, 필터링 기능 사용)

## 참고문헌

- 가속도계 기반 속도 추정 방법론
- 스마트워치 센서 데이터 처리 기법

## 라이선스

이 코드는 교육 및 연구 목적으로 작성되었습니다.
