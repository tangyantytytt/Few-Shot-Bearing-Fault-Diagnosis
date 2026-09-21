# PU/pu_dataset.py
import numpy as np
import os
from scipy.io import loadmat
import random

class PU:
    def __init__(self, split, exps=None, rpms=None, length=2048, data_path='Datasets/PU'):
        self.split = split
        self.length = length
        self.data_path = data_path
        
        # PU数据集官方13类故障类型映射
        self.faults_idx = {
            'K001': 0,    # 健康
            'K002': 0,    # 健康
            'K003': 0,    # 健康
            'K004': 0,    # 健康
            'K005': 0,    # 健康
            'K006': 0,    # 健康
            'KA01': 1,    # 外圈-电火花加工-1-人工
            'KA03': 2,    # 外圈-电刻蚀-2-人工
            'KA04': 3,    # 外圈-点蚀-1-真实
            'KA07': 4,    # 外圈-钻孔-1-人工
            'KA15': 5,    # 外圈-塑性变形-1-真实
            'KA16': 6,    # 外圈-点蚀-2-真实
            'KI01': 7,    # 内圈-电火花加工-2-人工
            'KI03': 8,    # 内圈-电刻蚀-1-人工
            'KI04': 9,    # 内圈-点蚀-1-真实
            'KI07': 10,   # 内圈-电刻蚀-2-人工
            'KI16': 11,   # 内圈-点蚀-3-真实
            'KI18': 12,   # 内圈-点蚀-2-真实
        }
        
        self._load_and_slice_data()
        self._shuffle()
        
        self.nclasses = len(np.unique(self.y_train))
        self.classes = sorted(list(set([(f'Class_{i}', i) for i in range(self.nclasses)])), 
                            key=lambda x: x[1])
        self.all_labels = self.classes

    def _get_fault_type(self, folder_name):
        """根据文件夹名判断故障类型（官方13类）"""
        if folder_name in self.faults_idx:
            return folder_name
        return None

    def _load_mat_file(self, file_path):
        """加载PU数据集的mat文件，返回双通道数据（force + phase_current_1）"""
        try:
            mat_dict = loadmat(file_path)
            
            data_keys = [k for k in mat_dict.keys() if not k.startswith('__')]
            
            for key in data_keys:
                try:
                    record = mat_dict[key][0, 0]
                    
                    if 'Y' in record.dtype.names:
                        y_data = record['Y']
                        
                        # 收集所有可用信号
                        signals = []
                        signal_names = []
                        
                        if isinstance(y_data, np.ndarray) and len(y_data) > 0:
                            for i in range(min(len(y_data), 5)):  # 最多取5个信号
                                try:
                                    signal_info = y_data[0, i]
                                    if 'Data' in signal_info.dtype.names:
                                        sig = signal_info['Data'].flatten()
                                        if len(sig) > 10000:
                                            signals.append(sig)
                                            if 'Name' in signal_info.dtype.names:
                                                name = str(signal_info['Name'].flatten()[0]) if len(signal_info['Name'].flatten()) > 0 else f'signal_{i}'
                                            else:
                                                name = f'signal_{i}'
                                            signal_names.append(name)
                                except:
                                    continue
                        
                        # 优先选择：force + phase_current_1（振动信号+电流信号）
                        if len(signals) >= 2:
                            # 查找force信号
                            force_idx = -1
                            current_idx = -1
                            
                            for idx, name in enumerate(signal_names):
                                if 'force' in name.lower() and force_idx == -1:
                                    force_idx = idx
                                elif 'current' in name.lower() and current_idx == -1:
                                    current_idx = idx
                            
                            if force_idx != -1 and current_idx != -1:
                                # 使用force和phase_current_1
                                return signals[force_idx], signals[current_idx]
                            else:
                                # 使用前两个信号
                                return signals[0], signals[1]
                        elif len(signals) == 1:
                            # 只有一个信号，复制
                            return signals[0], signals[0]
                    
                except Exception as e:
                    continue
            
            return None, None
            
        except Exception as e:
            return None, None

    def _load_and_slice_data(self):
        """加载并切分数据"""
        self.X_train = np.zeros((0, self.length, 2))
        self.X_test = np.zeros((0, self.length, 2))
        self.y_train = []
        self.y_test = []
        
        # PU信号约16008个点
        train_cuts = list(range(0, 10000, 80))[:self.split]
        test_cuts = list(range(10000, 16000, self.length))[:3]
        
        if not os.path.exists(self.data_path):
            print(f"数据路径 {self.data_path} 不存在！")
            return
        
        folders = [f for f in os.listdir(self.data_path) 
                  if os.path.isdir(os.path.join(self.data_path, f))]
        
        print(f"找到 {len(folders)} 个文件夹")
        
        for folder_name in sorted(folders):
            folder_path = os.path.join(self.data_path, folder_name)
            fault_type = self._get_fault_type(folder_name)
            
            if fault_type is None or fault_type not in self.faults_idx:
                print(f"跳过未识别文件夹: {folder_name}")
                continue
            
            mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
            
            for mat_file in mat_files:
                file_path = os.path.join(folder_path, mat_file)
                signal_ch1, signal_ch2 = self._load_mat_file(file_path)
                
                if signal_ch1 is None or signal_ch2 is None:
                    continue
                
                # 确保两个通道长度一致
                min_len = min(len(signal_ch1), len(signal_ch2))
                if min_len < self.length:
                    continue
                
                signal_ch1 = signal_ch1[:min_len]
                signal_ch2 = signal_ch2[:min_len]
                
                time_series = np.column_stack((signal_ch1, signal_ch2))
                
                for cut in train_cuts:
                    if cut + self.length <= time_series.shape[0]:
                        clip = time_series[cut:cut+self.length].reshape(1, self.length, 2)
                        self.X_train = np.vstack((self.X_train, clip))
                        self.y_train.append(self.faults_idx[fault_type])
                
                for cut in test_cuts:
                    if cut + self.length <= time_series.shape[0]:
                        clip = time_series[cut:cut+self.length].reshape(1, self.length, 2)
                        self.X_test = np.vstack((self.X_test, clip))
                        self.y_test.append(self.faults_idx[fault_type])
        
        print(f"训练数据形状: {self.X_train.shape}")
        print(f"测试数据形状: {self.X_test.shape}")
        if len(self.y_train) > 0:
            print(f"训练标签分布: {np.bincount(self.y_train)}")
        if len(self.y_test) > 0:
            print(f"测试标签分布: {np.bincount(self.y_test)}")

    def _shuffle(self):
        if len(self.y_train) > 0:
            index = list(range(self.X_train.shape[0]))
            random.Random(0).shuffle(index)
            self.X_train = self.X_train[index]
            self.y_train = np.array([self.y_train[i] for i in index])
        
        if len(self.y_test) > 0:
            index = list(range(self.X_test.shape[0]))
            random.Random(0).shuffle(index)
            self.X_test = self.X_test[index]
            self.y_test = np.array([self.y_test[i] for i in index])

    def add_noise_to_test_data(self, snr_dB, noise_amplitude=0.5):
        snr_linear = (snr_dB / 10.0)
        noise_power = 1.0 / snr_linear
        noise = noise_amplitude * np.sqrt(noise_power) * np.random.randn(*self.X_test.shape)
        self.X_test_noisy = self.X_test + noise