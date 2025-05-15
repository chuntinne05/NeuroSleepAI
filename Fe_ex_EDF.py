import numpy as np 
import pandas as pd 
import matplotlib as plt 
from scipy.fft import fft, fftfreq
import pywt
import mne
from tqdm import tqdm

def load_eeg_from_edf(file_path):
    print(f"Reading EDF file : {file_path}")
    raw = mne.io.read_raw_edf(file_path, preload=True)
    return raw

def extract_features_fft(signal, sample_rate):
    n = len(signal)
    yf = fft(signal)

    xf = fftfreq(n, 1/sample_rate)

    xf = xf[:n//2]
    yf_abs = 2.0/n * np.abs(yf[:n//2])

    delta_mask = (xf >= 0.5) & (xf < 4)
    theta_mask = (xf >= 4) & (xf < 8)
    alpha_mask = (xf >= 8) & (xf < 13)
    beta_mask = (xf >= 13) & (xf < 30)
    gamma_mask = (xf >= 30) & (xf < 100)

    delta_power = np.mean(yf_abs[delta_mask]) if np.any(delta_mask) else 0
    theta_power = np.mean(yf_abs[theta_mask]) if np.any(theta_mask) else 0
    alpha_power = np.mean(yf_abs[alpha_mask]) if np.any(alpha_mask) else 0
    beta_power = np.mean(yf_abs[beta_mask]) if np.any(beta_mask) else 0
    gamma_power = np.mean(yf_abs[gamma_mask]) if np.any(gamma_mask) else 0

    total_power = np.sum(yf_abs)
    if total_power > 0:
        delta_ratio = delta_power * np.sum(delta_mask) / total_power if np.any(delta_mask) else 0
        theta_ratio = theta_power * np.sum(theta_mask) / total_power if np.any(theta_mask) else 0
        alpha_ratio = alpha_power * np.sum(alpha_mask) / total_power if np.any(alpha_mask) else 0
        beta_ratio = beta_power * np.sum(beta_mask) / total_power if np.any(beta_mask) else 0
        gamma_ratio = gamma_power * np.sum(gamma_mask) / total_power if np.any(gamma_mask) else 0
    else:
        delta_ratio = theta_ratio = alpha_ratio = beta_ratio = gamma_ratio = 0
    
    peak_idx = np.argmax(yf_abs)
    peak_frequency = xf[peak_idx] if peak_idx < len(xf) else 0
    spectral_edge = np.percentile(xf, 95) if len(xf) > 0 else 0

    features = {
        'delta_power': delta_power,
        'theta_power': theta_power,
        'alpha_power': alpha_power,
        'beta_power': beta_power,
        'gamma_power': gamma_power,
        'delta_ratio': delta_ratio,
        'theta_ratio': theta_ratio,
        'alpha_ratio': alpha_ratio,
        'beta_ratio': beta_ratio,
        'gamma_ratio': gamma_ratio,
        'peak_frequency': peak_frequency,
        'spectral_edge': spectral_edge,
        'total_power': total_power
    }
    
    return features, xf, yf_abs

def extract_features_wavelet(signal, sample_rate, wavelet='db4', level=5):
    """
    Trích xuất đặc trưng từ tín hiệu EEG sử dụng biến đổi Wavelet
    
    Parameters:
    -----------
    signal : ndarray
        Tín hiệu EEG một chiều
    sample_rate : float
        Tần số lấy mẫu (Hz)
    wavelet : str
        Loại wavelet (mặc định: 'db4')
    level : int
        Số cấp độ phân tích (mặc định: 5)
        
    Returns:
    --------
    features : dict
        Dictionary chứa các đặc trưng wavelet
    coeffs : list
        Danh sách các hệ số wavelet
    """
    # Xác định số cấp độ tối đa có thể phân tích
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    level = min(level, max_level)
    
    # Thực hiện phân tích wavelet
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Trích xuất các hệ số từ các cấp khác nhau
    cA = coeffs[0]  # Hệ số xấp xỉ
    cD = coeffs[1:]  # Hệ số chi tiết
    
    features = {}
    
    # Tính năng lượng của hệ số xấp xỉ
    features['approx_energy'] = np.sum(cA**2) / len(cA) if len(cA) > 0 else 0
    
    # Tính năng lượng của các hệ số chi tiết
    for i, c in enumerate(cD):
        features[f'detail_{i+1}_energy'] = np.sum(c**2) / len(c) if len(c) > 0 else 0
    
    # Tính các đặc trưng thống kê cho mỗi cấp
    for i, c in enumerate(coeffs):
        if len(c) > 0:
            features[f'level_{i}_mean'] = np.mean(c)
            features[f'level_{i}_std'] = np.std(c)
            if np.std(c) > 0:
                features[f'level_{i}_kurtosis'] = np.mean((c - np.mean(c))**4) / (np.std(c)**4)
                features[f'level_{i}_skewness'] = np.mean((c - np.mean(c))**3) / (np.std(c)**3)
            else:
                features[f'level_{i}_kurtosis'] = 0
                features[f'level_{i}_skewness'] = 0
        else:
            features[f'level_{i}_mean'] = 0
            features[f'level_{i}_std'] = 0
            features[f'level_{i}_kurtosis'] = 0
            features[f'level_{i}_skewness'] = 0
    
    return features, coeffs

def extract_channel_features(signal, sample_rate, channel_name):
    time_features = {
        'channel': channel_name,
        'mean': np.mean(signal),
        'std': np.std(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'ptp': np.ptp(signal),  # Peak-to-peak amplitude
        'var': np.var(signal),
        'rms': np.sqrt(np.mean(signal**2)),
        'kurtosis': np.mean((signal - np.mean(signal))**4) / (np.std(signal)**4) if np.std(signal) > 0 else 0,
        'skewness': np.mean((signal - np.mean(signal))**3) / (np.std(signal)**3) if np.std(signal) > 0 else 0
    }
    
    # Trích xuất đặc trưng FFT
    fft_features, _, _ = extract_features_fft(signal, sample_rate)

    wavelet_features, _ = extract_features_wavelet(signal, sample_rate)
    
    # Kết hợp tất cả đặc trưng
    all_features = {**time_features, **fft_features, **wavelet_features}
    
    return all_features

def process_eeg_file(file_path, epoch_duration=1.0, overlap=0.5, selected_channels=None):

    raw = load_eeg_from_edf(file_path)
    
    # Lấy thông tin cơ bản
    sample_rate = raw.info['sfreq']
    channels = raw.ch_names
    
    if selected_channels is None:
        selected_channels = channels
    else:
        # Chỉ giữ lại các kênh có trong dữ liệu
        selected_channels = [ch for ch in selected_channels if ch in channels]
    
    if not selected_channels:
        raise ValueError("Không tìm thấy kênh nào phù hợp trong dữ liệu EEG")
    
    print(f"Tần số lấy mẫu: {sample_rate} Hz")
    print(f"Các kênh được chọn: {selected_channels}")
    
    # Tính số mẫu cho mỗi đoạn và bước nhảy
    epoch_samples = int(epoch_duration * sample_rate)
    step_samples = int(epoch_samples * (1 - overlap))
    
    # Chuẩn bị danh sách để lưu các đặc trưng
    all_features = []
    
    # Lấy dữ liệu
    data, times = raw.get_data(return_times=True)
    
    # Xử lý từng kênh
    for ch_idx, channel in enumerate(selected_channels):
        print(f"Đang xử lý kênh {channel}...")
        
        # Lấy dữ liệu kênh
        ch_idx_in_raw = channels.index(channel)
        channel_data = data[ch_idx_in_raw]
        
        # Chia thành các đoạn và trích xuất đặc trưng
        total_samples = len(channel_data)
        
        for start in tqdm(range(0, total_samples - epoch_samples, step_samples)):
            end = start + epoch_samples
            epoch_data = channel_data[start:end]
            
            # Tính thời gian
            epoch_time = times[start]
            
            # Trích xuất đặc trưng
            features = extract_channel_features(epoch_data, sample_rate, channel)
            
            # Thêm thông tin thời gian và epoch
            features['start_time'] = epoch_time
            features['epoch'] = start // step_samples
            
            all_features.append(features)
    
    # Tạo DataFrame
    features_df = pd.DataFrame(all_features)
    
    return features_df

def visualize_features(features_df, output_dir='.'):
    """
    Vẽ và lưu biểu đồ các đặc trưng quan trọng
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame chứa đặc trưng
    output_dir : str
        Thư mục đầu ra để lưu biểu đồ
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Vẽ biểu đồ năng lượng dải tần
    plt.figure(figsize=(12, 8))
    for channel in features_df['channel'].unique():
        channel_data = features_df[features_df['channel'] == channel]
        plt.plot(channel_data['epoch'], channel_data['alpha_power'], label=f'{channel} - Alpha')
        plt.plot(channel_data['epoch'], channel_data['beta_power'], label=f'{channel} - Beta')
        plt.plot(channel_data['epoch'], channel_data['theta_power'], label=f'{channel} - Theta')
        plt.plot(channel_data['epoch'], channel_data['delta_power'], label=f'{channel} - Delta')
    
    plt.title('Năng lượng các dải tần theo thời gian')
    plt.xlabel('Epoch')
    plt.ylabel('Năng lượng')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'band_powers.png'))
    
    # Vẽ biểu đồ tỷ lệ alpha/beta
    plt.figure(figsize=(12, 6))
    for channel in features_df['channel'].unique():
        channel_data = features_df[features_df['channel'] == channel]
        alpha_beta_ratio = channel_data['alpha_power'] / channel_data['beta_power']
        plt.plot(channel_data['epoch'], alpha_beta_ratio, label=channel)
    
    plt.title('Tỷ lệ Alpha/Beta theo thời gian')
    plt.xlabel('Epoch')
    plt.ylabel('Tỷ lệ Alpha/Beta')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'alpha_beta_ratio.png'))


def main(edf_file_path, output_csv_path, visualize=True, output_dir='.', 
        epoch_duration=1.0, overlap=0.5, selected_channels=None):

    print(f"Bắt đầu xử lý file: {edf_file_path}")
    
    # Trích xuất đặc trưng
    features_df = process_eeg_file(
        edf_file_path, 
        epoch_duration=epoch_duration,
        overlap=overlap,
        selected_channels=selected_channels
    )
    
    # Lưu đặc trưng vào file CSV
    features_df.to_csv(output_csv_path, index=False)
    print(f"Đã lưu đặc trưng vào: {output_csv_path}")
    
    # Hiển thị và lưu biểu đồ
    if visualize:
        visualize_features(features_df, output_dir)
        print(f"Đã lưu biểu đồ vào thư mục: {output_dir}")
    
    return features_df

# Ví dụ sử dụng
if __name__ == "__main__":
    # Thay đổi đường dẫn file EDF của bạn tại đây
    edf_file_path = "data/10000_17728.edf"
    output_csv_path = "eeg_features.csv"
    raw = load_eeg_from_edf(edf_file_path)
    print("Danh sach kenh : ", raw.ch_names)
    # Nếu biết tên các kênh cụ thể muốn phân tích, hãy liệt kê chúng ở đây
    # Nếu để None, tất cả các kênh sẽ được phân tích
    selected_channels = ['EEG Fpz-Cz']
    
    # Xử lý file EEG
    features_df = main(
        edf_file_path, 
        output_csv_path,
        visualize=True,
        output_dir='./eeg_plots',
        epoch_duration=1.0,  # Mỗi đoạn 1 giây
        overlap=0.5,  # Chồng lấp 50%
        selected_channels=selected_channels
    )
    
    # Hiển thị 5 dòng đầu tiên của DataFrame đặc trưng
    print("\nMẫu dữ liệu đặc trưng đã trích xuất:")
    print(features_df.head())