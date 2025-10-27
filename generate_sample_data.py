#!/usr/bin/env python3
"""
EMD 분석용 샘플 데이터 생성기
0.2Hz~4Hz 주파수 대역의 복합 신호를 생성하여 CSV 파일로 저장

Author: Signal Processing Tool
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_sample_signals():
    """샘플 신호 생성"""
    
    # 기본 설정
    fs = 1.0  # 샘플링 주파수 (1Hz, 1초 단위)
    duration = 3600  # 1시간 (3600초)
    t = np.arange(0, duration, 1/fs)
    
    print(f"샘플 데이터 생성 중...")
    print(f"샘플링 주파수: {fs}Hz")
    print(f"데이터 길이: {duration}초")
    print(f"총 샘플 수: {len(t)}")
    
    # 신호 1: 심박수 변동성 유사 신호 (0.5Hz ~ 2Hz)
    hrv_signal = (
        1.2 * np.sin(2 * np.pi * 0.8 * t) +  # 주요 성분 (0.8Hz)
        0.8 * np.sin(2 * np.pi * 1.5 * t) +  # 보조 성분 (1.5Hz)
        0.5 * np.sin(2 * np.pi * 0.3 * t) +  # 저주파 성분 (0.3Hz)
        0.3 * np.sin(2 * np.pi * 2.5 * t) +  # 고주파 성분 (2.5Hz)
        0.2 * np.random.randn(len(t))        # 노이즈
    )
    
    # 신호 2: 호흡 관련 신호 (0.2Hz ~ 0.5Hz)
    respiratory_signal = (
        2.0 * np.sin(2 * np.pi * 0.25 * t) + # 주요 호흡 성분 (0.25Hz, 15/min)
        0.8 * np.sin(2 * np.pi * 0.4 * t) +  # 보조 성분 (0.4Hz)
        0.5 * np.sin(2 * np.pi * 0.15 * t) + # 저주파 성분 (0.15Hz)
        0.3 * np.random.randn(len(t))        # 노이즈
    )
    
    # 신호 3: 뇌파 유사 신호 (1Hz ~ 4Hz)
    eeg_like_signal = (
        1.5 * np.sin(2 * np.pi * 2.0 * t) +  # 델타파 유사 (2Hz)
        1.0 * np.sin(2 * np.pi * 3.5 * t) +  # 세타파 유사 (3.5Hz)
        0.8 * np.sin(2 * np.pi * 1.2 * t) +  # 저주파 성분 (1.2Hz)
        0.6 * np.sin(2 * np.pi * 2.8 * t) +  # 중간 주파수 (2.8Hz)
        0.4 * np.random.randn(len(t))        # 노이즈
    )
    
    # 신호 4: 복합 생체 신호 (전체 대역 0.2Hz ~ 4Hz)
    complex_biosignal = (
        1.8 * np.sin(2 * np.pi * 0.3 * t + np.pi/4) +  # 0.3Hz 성분
        1.5 * np.sin(2 * np.pi * 1.0 * t + np.pi/3) +  # 1.0Hz 성분
        1.2 * np.sin(2 * np.pi * 2.2 * t + np.pi/6) +  # 2.2Hz 성분
        0.9 * np.sin(2 * np.pi * 3.7 * t + np.pi/2) +  # 3.7Hz 성분
        0.6 * np.sin(2 * np.pi * 0.8 * t) +            # 0.8Hz 성분
        0.4 * np.sin(2 * np.pi * 1.8 * t) +            # 1.8Hz 성분
        0.5 * np.random.randn(len(t))                   # 노이즈
    )
    
    # 신호 5: 시간에 따라 변하는 진폭 변조 신호
    amplitude_modulated = (
        (1 + 0.8 * np.sin(2 * np.pi * 0.01 * t)) *     # 진폭 변조 (0.01Hz)
        np.sin(2 * np.pi * 1.5 * t) +                   # 주파수 1.5Hz
        0.5 * np.sin(2 * np.pi * 0.4 * t) +            # 0.4Hz 성분
        0.3 * np.random.randn(len(t))                   # 노이즈
    )
    
    # 신호 6: 주파수 변조 신호
    frequency_modulated = (
        np.sin(2 * np.pi * (2.0 + 0.5 * np.sin(2 * np.pi * 0.02 * t)) * t) + # 주파수 변조
        0.8 * np.sin(2 * np.pi * 0.6 * t) +            # 0.6Hz 성분
        0.4 * np.random.randn(len(t))                   # 노이즈
    )
    
    # DataFrame 생성
    data = pd.DataFrame({
        'Time': t,
        'HRV_Signal': hrv_signal,
        'Respiratory_Signal': respiratory_signal,
        'EEG_Like_Signal': eeg_like_signal,
        'Complex_Biosignal': complex_biosignal,
        'Amplitude_Modulated': amplitude_modulated,
        'Frequency_Modulated': frequency_modulated
    })
    
    return data

def save_sample_data():
    """샘플 데이터 생성 및 저장"""
    
    # 데이터 생성
    data = generate_sample_signals()
    
    # CSV 파일로 저장
    filename = 'sample_emd_data.csv'
    data.to_csv(filename, index=False)
    print(f"\n샘플 데이터가 '{filename}'에 저장되었습니다.")
    
    # 데이터 정보 출력
    print(f"\n데이터 정보:")
    print(f"- 데이터 크기: {data.shape}")
    print(f"- 컬럼: {list(data.columns)}")
    print(f"- 시간 범위: {data['Time'].min():.1f}초 ~ {data['Time'].max():.1f}초")
    
    # 각 신호의 기본 통계
    print(f"\n각 신호의 기본 통계:")
    for col in data.columns:
        if col != 'Time':
            mean_val = data[col].mean()
            std_val = data[col].std()
            min_val = data[col].min()
            max_val = data[col].max()
            print(f"- {col}: 평균={mean_val:.3f}, 표준편차={std_val:.3f}, 범위=[{min_val:.3f}, {max_val:.3f}]")
    
    return data

def visualize_sample_data(data):
    """샘플 데이터 시각화"""
    
    # 시간축 (처음 300초만 표시)
    time_subset = data['Time'] <= 300
    t_plot = data.loc[time_subset, 'Time']
    
    # 플롯 생성
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('샘플 EMD 분석 데이터 (처음 300초)', fontsize=14)
    
    signal_columns = [col for col in data.columns if col != 'Time']
    
    for i, col in enumerate(signal_columns):
        row = i // 2
        col_idx = i % 2
        
        axes[row, col_idx].plot(t_plot, data.loc[time_subset, col])
        axes[row, col_idx].set_title(col.replace('_', ' '))
        axes[row, col_idx].set_xlabel('시간 (초)')
        axes[row, col_idx].set_ylabel('진폭')
        axes[row, col_idx].grid(True)
    
    plt.tight_layout()
    plt.savefig('sample_data_preview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("샘플 데이터 미리보기가 'sample_data_preview.png'에 저장되었습니다.")

def create_frequency_analysis():
    """주파수 분석 미리보기"""
    
    data = pd.read_csv('sample_emd_data.csv')
    
    # 주파수 분석
    from scipy.signal import welch
    
    fs = 1.0  # 샘플링 주파수
    signal_columns = [col for col in data.columns if col != 'Time']
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('샘플 데이터 주파수 스펙트럼', fontsize=14)
    
    for i, col in enumerate(signal_columns):
        row = i // 2
        col_idx = i % 2
        
        # Welch 방법으로 파워 스펙트럼 계산
        freqs, psd = welch(data[col], fs=fs, nperseg=min(1024, len(data)//4))
        
        axes[row, col_idx].semilogy(freqs, psd)
        axes[row, col_idx].set_title(f'{col.replace("_", " ")} 파워 스펙트럼')
        axes[row, col_idx].set_xlabel('주파수 (Hz)')
        axes[row, col_idx].set_ylabel('파워')
        axes[row, col_idx].grid(True)
        axes[row, col_idx].set_xlim([0, 5])  # 0-5Hz 범위 표시
        
        # 분석 대상 주파수 대역 표시 (0.2-4Hz)
        axes[row, col_idx].axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label='분석 대역')
        axes[row, col_idx].axvline(x=4.0, color='red', linestyle='--', alpha=0.7)
        axes[row, col_idx].legend()
    
    plt.tight_layout()
    plt.savefig('sample_data_frequency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("주파수 분석 결과가 'sample_data_frequency_analysis.png'에 저장되었습니다.")

if __name__ == "__main__":
    print("EMD 분석용 샘플 데이터 생성기")
    print("=" * 50)
    
    # 샘플 데이터 생성 및 저장
    data = save_sample_data()
    
    # 데이터 시각화
    print("\n데이터 시각화 중...")
    visualize_sample_data(data)
    
    # 주파수 분석
    print("\n주파수 분석 중...")
    create_frequency_analysis()
    
    print("\n" + "=" * 50)
    print("샘플 데이터 생성 완료!")
    print("이제 'python emd_analysis_tool.py'를 실행하여 분석 도구를 사용할 수 있습니다.")
    print("생성된 'sample_emd_data.csv' 파일을 분석 도구에서 불러와 테스트해보세요.")