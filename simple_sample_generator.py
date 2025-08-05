#!/usr/bin/env python3
"""
간단한 EMD 분석용 샘플 데이터 생성기
"""

import numpy as np
import pandas as pd

def generate_simple_sample():
    """간단한 샘플 데이터 생성"""
    
    # 기본 설정 (더 작은 데이터셋)
    fs = 1.0  # 샘플링 주파수 (1Hz, 1초 단위)
    duration = 600  # 10분 (600초)
    t = np.arange(0, duration, 1/fs)
    
    print(f"샘플 데이터 생성 중...")
    print(f"데이터 길이: {duration}초, 샘플 수: {len(t)}")
    
    # 신호 1: 심박수 변동성 유사 신호 (0.5Hz ~ 2Hz)
    hrv_signal = (
        1.2 * np.sin(2 * np.pi * 0.8 * t) +  # 주요 성분 (0.8Hz)
        0.8 * np.sin(2 * np.pi * 1.5 * t) +  # 보조 성분 (1.5Hz)
        0.5 * np.sin(2 * np.pi * 0.3 * t) +  # 저주파 성분 (0.3Hz)
        0.2 * np.random.randn(len(t))        # 노이즈
    )
    
    # 신호 2: 호흡 관련 신호 (0.2Hz ~ 0.5Hz)
    respiratory_signal = (
        2.0 * np.sin(2 * np.pi * 0.25 * t) + # 주요 호흡 성분 (0.25Hz)
        0.8 * np.sin(2 * np.pi * 0.4 * t) +  # 보조 성분 (0.4Hz)
        0.3 * np.random.randn(len(t))        # 노이즈
    )
    
    # 신호 3: 복합 생체 신호 (전체 대역 0.2Hz ~ 4Hz)
    complex_biosignal = (
        1.8 * np.sin(2 * np.pi * 0.3 * t) +  # 0.3Hz 성분
        1.5 * np.sin(2 * np.pi * 1.0 * t) +  # 1.0Hz 성분
        1.2 * np.sin(2 * np.pi * 2.2 * t) +  # 2.2Hz 성분
        0.9 * np.sin(2 * np.pi * 3.7 * t) +  # 3.7Hz 성분
        0.5 * np.random.randn(len(t))        # 노이즈
    )
    
    # DataFrame 생성
    data = pd.DataFrame({
        'Time': t,
        'HRV_Signal': hrv_signal,
        'Respiratory_Signal': respiratory_signal,
        'Complex_Biosignal': complex_biosignal
    })
    
    return data

if __name__ == "__main__":
    print("간단한 EMD 분석용 샘플 데이터 생성기")
    print("=" * 50)
    
    # 샘플 데이터 생성
    data = generate_simple_sample()
    
    # CSV 파일로 저장
    filename = 'sample_emd_data.csv'
    data.to_csv(filename, index=False)
    print(f"샘플 데이터가 '{filename}'에 저장되었습니다.")
    
    # 데이터 정보 출력
    print(f"\n데이터 정보:")
    print(f"- 데이터 크기: {data.shape}")
    print(f"- 컬럼: {list(data.columns)}")
    print(f"- 시간 범위: {data['Time'].min():.1f}초 ~ {data['Time'].max():.1f}초")
    
    print("\n완료! 이제 'python3 emd_analysis_tool.py'를 실행하여 분석 도구를 사용할 수 있습니다.")