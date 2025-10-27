#!/usr/bin/env python3
"""
EMD, EEMD, FIF 분석 도구
0.2Hz~4Hz 주파수 대역을 대상으로 한 신호 분석
CSV 파일의 컬럼을 선택하여 분석 가능

Author: Signal Processing Tool
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import hilbert, butter, filtfilt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import warnings
warnings.filterwarnings('ignore')

class EMDAnalysisTool:
    def __init__(self):
        self.data = None
        self.fs = 1.0  # 기본 샘플링 주파수 (1Hz, 1초 단위)
        self.time_vector = None
        self.selected_column = None
        self.analysis_results = {}
        
        # 주파수 대역 설정 (0.2Hz ~ 4Hz)
        self.freq_low = 0.2
        self.freq_high = 4.0
        
        # GUI 설정
        self.setup_gui()
    
    def setup_gui(self):
        """GUI 설정"""
        self.root = tk.Tk()
        self.root.title("EMD/EEMD/FIF 분석 도구 (0.2Hz~4Hz)")
        self.root.geometry("800x600")
        
        # 메인 프레임
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 파일 선택 프레임
        file_frame = ttk.LabelFrame(main_frame, text="데이터 파일 선택")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="CSV 파일 선택", 
                  command=self.load_csv_file).pack(side=tk.LEFT, padx=5, pady=5)
        
        self.file_label = ttk.Label(file_frame, text="파일이 선택되지 않음")
        self.file_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # 설정 프레임
        settings_frame = ttk.LabelFrame(main_frame, text="분석 설정")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 컬럼 선택
        ttk.Label(settings_frame, text="분석할 컬럼:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.column_var = tk.StringVar()
        self.column_combo = ttk.Combobox(settings_frame, textvariable=self.column_var, 
                                        state="readonly", width=20)
        self.column_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # 샘플링 주파수
        ttk.Label(settings_frame, text="샘플링 주파수 (Hz):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.fs_var = tk.StringVar(value="1.0")
        ttk.Entry(settings_frame, textvariable=self.fs_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # 주파수 대역 설정
        ttk.Label(settings_frame, text="주파수 대역:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        freq_frame = ttk.Frame(settings_frame)
        freq_frame.grid(row=1, column=1, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        self.freq_low_var = tk.StringVar(value="0.2")
        self.freq_high_var = tk.StringVar(value="4.0")
        ttk.Entry(freq_frame, textvariable=self.freq_low_var, width=8).pack(side=tk.LEFT)
        ttk.Label(freq_frame, text=" ~ ").pack(side=tk.LEFT)
        ttk.Entry(freq_frame, textvariable=self.freq_high_var, width=8).pack(side=tk.LEFT)
        ttk.Label(freq_frame, text=" Hz").pack(side=tk.LEFT)
        
        # 분석 방법 선택
        method_frame = ttk.LabelFrame(main_frame, text="분석 방법 선택")
        method_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.emd_var = tk.BooleanVar(value=True)
        self.eemd_var = tk.BooleanVar(value=True)
        self.fif_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(method_frame, text="EMD", variable=self.emd_var).pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Checkbutton(method_frame, text="EEMD", variable=self.eemd_var).pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Checkbutton(method_frame, text="FIF", variable=self.fif_var).pack(side=tk.LEFT, padx=10, pady=5)
        
        # 분석 실행 버튼
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="분석 실행", 
                  command=self.run_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="결과 저장", 
                  command=self.save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="종료", 
                  command=self.root.quit).pack(side=tk.RIGHT, padx=5)
        
        # 진행 상태 표시
        self.progress_var = tk.StringVar(value="대기 중...")
        self.progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        self.progress_label.pack(pady=5)
        
        # 결과 표시 영역
        result_frame = ttk.LabelFrame(main_frame, text="분석 결과")
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = tk.Text(result_frame, height=15, width=80)
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def load_csv_file(self):
        """CSV 파일 로드"""
        file_path = filedialog.askopenfilename(
            title="CSV 파일 선택",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.file_label.config(text=f"파일: {os.path.basename(file_path)}")
                
                # 컬럼 목록 업데이트
                numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
                self.column_combo['values'] = numeric_columns
                
                if numeric_columns:
                    self.column_combo.set(numeric_columns[0])
                
                self.update_result_text(f"CSV 파일 로드 완료: {file_path}\n")
                self.update_result_text(f"데이터 크기: {self.data.shape}\n")
                self.update_result_text(f"숫자형 컬럼: {numeric_columns}\n\n")
                
            except Exception as e:
                messagebox.showerror("오류", f"파일 로드 중 오류 발생:\n{str(e)}")
    
    def update_result_text(self, text):
        """결과 텍스트 업데이트"""
        self.result_text.insert(tk.END, text)
        self.result_text.see(tk.END)
        self.root.update()
    
    def run_analysis(self):
        """분석 실행"""
        if self.data is None:
            messagebox.showerror("오류", "먼저 CSV 파일을 선택해주세요.")
            return
        
        if not self.column_var.get():
            messagebox.showerror("오류", "분석할 컬럼을 선택해주세요.")
            return
        
        try:
            # 설정 값 가져오기
            self.fs = float(self.fs_var.get())
            self.freq_low = float(self.freq_low_var.get())
            self.freq_high = float(self.freq_high_var.get())
            self.selected_column = self.column_var.get()
            
            # 데이터 준비
            signal_data = self.data[self.selected_column].dropna().values
            self.time_vector = np.arange(len(signal_data)) / self.fs
            
            self.update_result_text("="*50 + "\n")
            self.update_result_text("EMD/EEMD/FIF 분석 시작\n")
            self.update_result_text("="*50 + "\n")
            self.update_result_text(f"선택된 컬럼: {self.selected_column}\n")
            self.update_result_text(f"데이터 길이: {len(signal_data)} 샘플\n")
            self.update_result_text(f"분석 시간: {len(signal_data)/self.fs:.2f} 초\n")
            self.update_result_text(f"주파수 대역: {self.freq_low}Hz ~ {self.freq_high}Hz\n\n")
            
            # 주파수 필터링
            filtered_signal = self.bandpass_filter(signal_data, self.freq_low, self.freq_high, self.fs)
            
            # 선택된 방법들로 분석 수행
            if self.emd_var.get():
                self.progress_var.set("EMD 분석 중...")
                self.run_emd_analysis(filtered_signal)
            
            if self.eemd_var.get():
                self.progress_var.set("EEMD 분석 중...")
                self.run_eemd_analysis(filtered_signal)
            
            if self.fif_var.get():
                self.progress_var.set("FIF 분석 중...")
                self.run_fif_analysis(filtered_signal)
            
            # 결과 시각화
            self.progress_var.set("결과 시각화 중...")
            self.visualize_results(filtered_signal)
            
            self.progress_var.set("분석 완료!")
            self.update_result_text("\n분석이 완료되었습니다!\n")
            
        except Exception as e:
            messagebox.showerror("오류", f"분석 중 오류 발생:\n{str(e)}")
            self.progress_var.set("오류 발생")
    
    def bandpass_filter(self, data, low_freq, high_freq, fs):
        """밴드패스 필터 적용"""
        nyquist = fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # 주파수가 나이퀴스트 주파수를 초과하지 않도록 제한
        high = min(high, 0.99)
        
        b, a = butter(4, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    def run_emd_analysis(self, signal_data):
        """EMD 분석 수행"""
        self.update_result_text("EMD 분석 수행 중...\n")
        
        try:
            imfs, residual = self.emd_decomposition(signal_data)
            
            # Hilbert 분석
            hilbert_results = self.hilbert_analysis(imfs, self.fs)
            
            self.analysis_results['emd'] = {
                'imfs': imfs,
                'residual': residual,
                'hilbert': hilbert_results,
                'num_imfs': len(imfs)
            }
            
            self.update_result_text(f"EMD 완료: {len(imfs)}개 IMF 추출\n")
            
            # 각 IMF의 주파수 특성 출력
            for i, result in enumerate(hilbert_results):
                mean_freq = np.mean(result['frequency'][result['frequency'] > 0])
                energy = result['energy']
                self.update_result_text(f"  IMF {i+1}: 평균 주파수 {mean_freq:.3f}Hz, 에너지 {energy:.6f}\n")
            
        except Exception as e:
            self.update_result_text(f"EMD 분석 오류: {str(e)}\n")
    
    def run_eemd_analysis(self, signal_data):
        """EEMD 분석 수행"""
        self.update_result_text("\nEEMD 분석 수행 중...\n")
        
        try:
            imfs, residual = self.eemd_decomposition(signal_data, num_ensembles=50)
            
            # Hilbert 분석
            hilbert_results = self.hilbert_analysis(imfs, self.fs)
            
            self.analysis_results['eemd'] = {
                'imfs': imfs,
                'residual': residual,
                'hilbert': hilbert_results,
                'num_imfs': len(imfs)
            }
            
            self.update_result_text(f"EEMD 완료: {len(imfs)}개 IMF 추출\n")
            
            # 각 IMF의 주파수 특성 출력
            for i, result in enumerate(hilbert_results):
                mean_freq = np.mean(result['frequency'][result['frequency'] > 0])
                energy = result['energy']
                self.update_result_text(f"  IMF {i+1}: 평균 주파수 {mean_freq:.3f}Hz, 에너지 {energy:.6f}\n")
            
        except Exception as e:
            self.update_result_text(f"EEMD 분석 오류: {str(e)}\n")
    
    def run_fif_analysis(self, signal_data):
        """FIF 분석 수행"""
        self.update_result_text("\nFIF 분석 수행 중...\n")
        
        try:
            imfs, residual = self.fif_decomposition(signal_data)
            
            # Hilbert 분석
            hilbert_results = self.hilbert_analysis(imfs, self.fs)
            
            self.analysis_results['fif'] = {
                'imfs': imfs,
                'residual': residual,
                'hilbert': hilbert_results,
                'num_imfs': len(imfs)
            }
            
            self.update_result_text(f"FIF 완료: {len(imfs)}개 IMF 추출\n")
            
            # 각 IMF의 주파수 특성 출력
            for i, result in enumerate(hilbert_results):
                mean_freq = np.mean(result['frequency'][result['frequency'] > 0])
                energy = result['energy']
                self.update_result_text(f"  IMF {i+1}: 평균 주파수 {mean_freq:.3f}Hz, 에너지 {energy:.6f}\n")
            
        except Exception as e:
            self.update_result_text(f"FIF 분석 오류: {str(e)}\n")
    
    def emd_decomposition(self, signal_data, max_imf=8):
        """EMD 분해"""
        imfs = []
        residual = signal_data.copy()
        
        for imf_idx in range(max_imf):
            h = residual.copy()
            
            # Sifting 과정
            for sift_iter in range(10):
                # 극값 찾기
                max_peaks = self.find_peaks(h)
                min_peaks = self.find_peaks(-h)
                
                if len(max_peaks) < 2 or len(min_peaks) < 2:
                    break
                
                # 스플라인 보간으로 포락선 생성
                t_vec = np.arange(len(h))
                
                try:
                    upper_env = interp1d(max_peaks, h[max_peaks], kind='cubic', 
                                       bounds_error=False, fill_value='extrapolate')(t_vec)
                    lower_env = interp1d(min_peaks, -(-h)[min_peaks], kind='cubic', 
                                       bounds_error=False, fill_value='extrapolate')(t_vec)
                except:
                    break
                
                # 평균 제거
                mean_env = (upper_env + lower_env) / 2
                new_h = h - mean_env
                
                # 수렴 조건 확인
                if np.sum((h - new_h)**2) / np.sum(h**2) < 0.01:
                    break
                
                h = new_h
            
            imfs.append(h)
            residual = residual - h
            
            # 종료 조건
            if np.sum(np.abs(residual)) < 0.01 * np.sum(np.abs(signal_data)):
                break
        
        return imfs, residual
    
    def eemd_decomposition(self, signal_data, num_ensembles=50, noise_std=0.1):
        """EEMD 분해"""
        N = len(signal_data)
        ensemble_imfs = []
        ensemble_residuals = []
        
        for ensemble in range(num_ensembles):
            # 백색 잡음 추가
            noisy_signal = signal_data + noise_std * np.random.randn(N)
            
            # EMD 분해
            imfs, residual = self.emd_decomposition(noisy_signal)
            ensemble_imfs.append(imfs)
            ensemble_residuals.append(residual)
            
            if (ensemble + 1) % 10 == 0:
                self.update_result_text(f"  EEMD 진행률: {ensemble + 1}/{num_ensembles}\n")
        
        # 앙상블 평균
        max_imfs = max(len(imfs) for imfs in ensemble_imfs)
        averaged_imfs = []
        
        for i in range(max_imfs):
            imf_sum = np.zeros(N)
            count = 0
            for imfs in ensemble_imfs:
                if i < len(imfs):
                    imf_sum += imfs[i]
                    count += 1
            if count > 0:
                averaged_imfs.append(imf_sum / count)
        
        averaged_residual = np.mean(ensemble_residuals, axis=0)
        
        return averaged_imfs, averaged_residual
    
    def fif_decomposition(self, signal_data, max_imf=8):
        """FIF 분해"""
        imfs = []
        f = signal_data.copy()
        N = len(f)
        
        # 마스크 길이 계산
        mask_lengths = np.unique(np.round(np.logspace(np.log10(2), np.log10(N//5), max_imf)).astype(int))
        
        for imf_idx in range(max_imf):
            if imf_idx < len(mask_lengths):
                mask_length = mask_lengths[imf_idx]
            else:
                mask_length = mask_lengths[-1]
            
            # 마스크 생성
            mask = np.ones(mask_length) / mask_length
            
            h = f.copy()
            
            # 반복 필터링
            for inner in range(200):
                h_old = h.copy()
                
                # 확장된 신호에 필터 적용
                h_ext = self.extend_signal(h, 3)
                h_filtered = np.convolve(h_ext, mask, mode='same')
                h_filtered = h_filtered[3:-3]
                
                # IMF 업데이트
                h = h - h_filtered
                
                # 수렴 조건 확인
                if np.linalg.norm(h - h_old) / np.linalg.norm(h_old) < 0.001:
                    break
            
            imfs.append(h)
            f = f - h
            
            # 종료 조건 확인
            if np.max(np.abs(f)) < 0.001 * np.max(np.abs(imfs[0])):
                break
        
        return imfs, f
    
    def find_peaks(self, data):
        """피크 찾기"""
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append(i)
        return np.array(peaks)
    
    def extend_signal(self, signal_data, ext_points):
        """신호 확장"""
        left_ext = signal_data[ext_points:0:-1]
        right_ext = signal_data[-2:-2-ext_points:-1]
        return np.concatenate([left_ext, signal_data, right_ext])
    
    def hilbert_analysis(self, imfs, fs):
        """Hilbert 변환 분석"""
        results = []
        
        for imf in imfs:
            # Hilbert 변환
            analytic_signal = hilbert(imf)
            
            # 순간 진폭 및 위상
            instantaneous_amplitude = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            
            # 순간 주파수
            instantaneous_frequency = fs / (2 * np.pi) * np.diff(np.concatenate([[instantaneous_phase[0]], instantaneous_phase]))
            
            # 에너지 계산
            energy = np.sum(instantaneous_amplitude**2) / len(imf)
            
            results.append({
                'amplitude': instantaneous_amplitude,
                'frequency': instantaneous_frequency,
                'energy': energy
            })
        
        return results
    
    def visualize_results(self, original_signal):
        """결과 시각화"""
        if not self.analysis_results:
            return
        
        # 시간축 (1초 단위)
        time_seconds = self.time_vector
        
        # 전체 결과 비교 플롯
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'EMD/EEMD/FIF 분석 결과 ({self.freq_low}Hz~{self.freq_high}Hz)', fontsize=14)
        
        methods = ['emd', 'eemd', 'fif']
        method_names = ['EMD', 'EEMD', 'FIF']
        
        for i, (method, method_name) in enumerate(zip(methods, method_names)):
            if method not in self.analysis_results:
                continue
            
            result = self.analysis_results[method]
            imfs = result['imfs']
            hilbert_results = result['hilbert']
            
            # IMF 시각화 (처음 3개만)
            ax1 = axes[0, i]
            for j, imf in enumerate(imfs[:3]):
                ax1.plot(time_seconds, imf, label=f'IMF {j+1}', alpha=0.7)
            ax1.set_title(f'{method_name} IMFs')
            ax1.set_xlabel('시간 (초)')
            ax1.set_ylabel('진폭')
            ax1.legend()
            ax1.grid(True)
            
            # 에너지 분포
            ax2 = axes[1, i]
            energies = [h['energy'] for h in hilbert_results]
            ax2.bar(range(1, len(energies)+1), energies)
            ax2.set_title(f'{method_name} 에너지 분포')
            ax2.set_xlabel('IMF 번호')
            ax2.set_ylabel('에너지')
            ax2.grid(True)
            
            # Hilbert 스펙트럼
            ax3 = axes[2, i]
            for j, h_result in enumerate(hilbert_results[:3]):
                freq = h_result['frequency']
                amp = h_result['amplitude']
                # 유효한 주파수 범위만 표시
                valid_idx = (freq > 0) & (freq < self.fs/2)
                if np.any(valid_idx):
                    scatter = ax3.scatter(freq[valid_idx], time_seconds[valid_idx], 
                                        c=amp[valid_idx], s=1, alpha=0.6, 
                                        label=f'IMF {j+1}')
            ax3.set_title(f'{method_name} Hilbert 스펙트럼')
            ax3.set_xlabel('순간 주파수 (Hz)')
            ax3.set_ylabel('시간 (초)')
            ax3.set_xlim([self.freq_low, self.freq_high])
            
            # 주파수 대역 표시
            rect = patches.Rectangle((self.freq_low, 0), self.freq_high - self.freq_low, 
                                   time_seconds[-1], linewidth=2, edgecolor='red', 
                                   facecolor='none', linestyle='--')
            ax3.add_patch(rect)
        
        plt.tight_layout()
        plt.show()
        
        # 개별 상세 분석 플롯
        self.plot_detailed_analysis(original_signal)
    
    def plot_detailed_analysis(self, original_signal):
        """상세 분석 플롯"""
        time_seconds = self.time_vector
        
        for method, method_name in [('emd', 'EMD'), ('eemd', 'EEMD'), ('fif', 'FIF')]:
            if method not in self.analysis_results:
                continue
            
            result = self.analysis_results[method]
            imfs = result['imfs']
            hilbert_results = result['hilbert']
            
            # 상세 IMF 분석
            fig, axes = plt.subplots(len(imfs) + 2, 1, figsize=(12, 2*(len(imfs)+2)))
            fig.suptitle(f'{method_name} 상세 분석 결과', fontsize=14)
            
            # 원본 신호
            axes[0].plot(time_seconds, original_signal)
            axes[0].set_title('원본 신호 (필터링됨)')
            axes[0].set_ylabel('진폭')
            axes[0].grid(True)
            
            # 각 IMF
            for i, (imf, h_result) in enumerate(zip(imfs, hilbert_results)):
                mean_freq = np.mean(h_result['frequency'][h_result['frequency'] > 0])
                energy = h_result['energy']
                
                axes[i+1].plot(time_seconds, imf)
                axes[i+1].set_title(f'IMF {i+1} (평균 주파수: {mean_freq:.3f}Hz, 에너지: {energy:.6f})')
                axes[i+1].set_ylabel('진폭')
                axes[i+1].grid(True)
            
            # 잔여 성분
            axes[-1].plot(time_seconds, result['residual'])
            axes[-1].set_title('잔여 성분')
            axes[-1].set_xlabel('시간 (초)')
            axes[-1].set_ylabel('진폭')
            axes[-1].grid(True)
            
            plt.tight_layout()
            plt.show()
    
    def save_results(self):
        """결과 저장"""
        if not self.analysis_results:
            messagebox.showwarning("경고", "저장할 결과가 없습니다.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # 결과를 DataFrame으로 정리
                results_data = {'Time': self.time_vector}
                
                for method in ['emd', 'eemd', 'fif']:
                    if method in self.analysis_results:
                        result = self.analysis_results[method]
                        imfs = result['imfs']
                        
                        for i, imf in enumerate(imfs):
                            results_data[f'{method.upper()}_IMF{i+1}'] = imf
                        
                        results_data[f'{method.upper()}_Residual'] = result['residual']
                
                df = pd.DataFrame(results_data)
                df.to_csv(file_path, index=False)
                
                messagebox.showinfo("완료", f"결과가 저장되었습니다:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("오류", f"저장 중 오류 발생:\n{str(e)}")
    
    def run(self):
        """GUI 실행"""
        self.root.mainloop()

if __name__ == "__main__":
    app = EMDAnalysisTool()
    app.run()