import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import train_test_split

class OutlierStdRemove(BaseEstimator, TransformerMixin):
    """
    Loại bỏ các hàng có outlier dựa trên phương pháp Độ lệch chuẩn.
    Phiên bản này được tối ưu hóa bằng phương pháp vectorized.

    Parameters
    ----------
    columns : list
        Danh sách tên các cột số cần kiểm tra outlier.
    factor : float, default=3.0
        Hệ số nhân với độ lệch chuẩn để xác định ngưỡng.
    """
    def __init__(self, columns, factor=3.0):
        self.columns = columns
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_subset = X[self.columns]
        
        # Tính toán vectorized
        means = X_subset.mean()
        stds = X_subset.std()
        lower_bounds = means - (stds * self.factor)
        upper_bounds = means + (stds * self.factor)
        
        self.bounds_ = {
            col: (lower_bounds[col], upper_bounds[col]) for col in self.columns
        }
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns:
            lower, upper = self.bounds_[col]
            X_transformed = X_transformed[
                (X_transformed[col] >= lower) & (X_transformed[col] <= upper)
            ]
        return X_transformed

class OutlierIQRRemove(BaseEstimator, TransformerMixin):
    """
    Loại bỏ các hàng có outlier dựa trên phương pháp Khoảng tứ phân vị (IQR).
    Phiên bản này được tối ưu hóa bằng phương pháp vectorized.

    Parameters
    ----------
    columns : list
        Danh sách tên các cột số cần kiểm tra outlier.
    factor : float, default=1.5
        Hệ số nhân với IQR để xác định ngưỡng.
    """
    def __init__(self, columns, factor=1.5):
        self.columns = columns
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_subset = X[self.columns]
        
        # Tính toán vectorized
        Q1 = X_subset.quantile(0.25)
        Q3 = X_subset.quantile(0.75)
        IQR = Q3 - Q1
        lower_bounds = Q1 - (IQR * self.factor)
        upper_bounds = Q3 + (IQR * self.factor)
        
        self.bounds_ = {
            col: (lower_bounds[col], upper_bounds[col]) for col in self.columns
        }
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns:
            lower, upper = self.bounds_[col]
            X_transformed = X_transformed[
                (X_transformed[col] >= lower) & (X_transformed[col] <= upper)
            ]
        return X_transformed

class ModelComparer:
    """
    So sánh hiệu suất của mô hình trên dữ liệu gốc và dữ liệu đã xử lý.
    Class này tuân thủ quy trình chống Data Leakage bằng cách chia train/test trước tiên.
    Nó có thể nhận một preprocessor đơn lẻ hoặc cả một Pipeline.
    """
    def __init__(self, model, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
        self.results_ = None
        
    def compare(self, X, y, test_size=0.2, random_state=42):
        # 1. Luôn chia train/test trước tiên để tránh data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 2. Huấn luyện và đánh giá trên dữ liệu RAW
        raw_model = clone(self.model)
        raw_model.fit(X_train, y_train)
        raw_score = raw_model.score(X_test, y_test)
        
        results = {
            'Dataset': ['Raw'], 'Model Score': [raw_score],
            'Train Samples': [len(X_train)], 'Test Samples': [len(X_test)]
        }

        # 3. Huấn luyện và đánh giá trên dữ liệu đã được xử lý (nếu có)
        if self.preprocessor:
            processor = clone(self.preprocessor)
            # Fit bộ tiền xử lý CHỈ trên X_train
            processor.fit(X_train)
            
            # Transform cả X_train và X_test
            X_train_processed = processor.transform(X_train)
            X_test_processed = processor.transform(X_test)
            
            # Đồng bộ y_train/y_test với X đã được xử lý (vì số hàng có thể thay đổi)
            y_train_processed = y_train.loc[X_train_processed.index]
            y_test_processed = y_test.loc[X_test_processed.index]
            
            # Huấn luyện mô hình trên dữ liệu sạch
            processed_model = clone(self.model)
            processed_model.fit(X_train_processed, y_train_processed)
            processed_score = processed_model.score(X_test_processed, y_test_processed)
            
            results['Dataset'].append('Processed')
            results['Model Score'].append(processed_score)
            results['Train Samples'].append(len(X_train_processed))
            results['Test Samples'].append(len(X_test_processed))
        
        self.results_ = pd.DataFrame(results).set_index('Dataset')
        return self.results_
