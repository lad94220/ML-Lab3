import pandas as pd
import pickle
import models
import numpy as np
import torch

INPUT_CSV_FILE = 'data/protein.csv' 
MODEL_PKL_FILE = 'FINAL_DEPLOYMENT_MODEL_protein_MAE_20251213_230400.pkl'
OUTPUT_CSV_FILE = 'MAE_PROTEIN_PREDICTIONS.csv'
TARGET_COLUMN = 'RMSD' 
SCALER_PKL_FILE = 'data/protein_scaler.pkl'

# --------------------------------------------------------------------------
## BƯỚC 1: ĐỌC DỮ LIỆU ĐẦU VÀO VÀ TÁCH TÍNH NĂNG (FEATURES)
# --------------------------------------------------------------------------
try:
    # Đọc toàn bộ dữ liệu từ file CSV. 
    # Vì dữ liệu bạn cung cấp không có index, nên dùng header=0 (dòng đầu là tên cột)
    data = pd.read_csv(INPUT_CSV_FILE)
    
    print(f"Đã đọc {len(data)} dòng dữ liệu từ file: {INPUT_CSV_FILE}")
    print(f"Các cột dữ liệu: {data.columns.tolist()}")
    X = data.drop(columns=[TARGET_COLUMN]) 
    
    # Lấy cột Ground Truth (Y)
    Y_ground_truth = data[TARGET_COLUMN].copy() 

    # Chuyển Features thành mảng numpy (thường cần cho mô hình ML)
    X_features = X.values 
    
    # In thông tin kiểm tra
    print(f"\nKích thước Features (X): {X_features.shape}")
    print(f"Kích thước Ground Truth (Y): {Y_ground_truth.shape}")

except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file dữ liệu đầu vào '{INPUT_CSV_FILE}'. Vui lòng kiểm tra lại đường dẫn.")
    exit()
except KeyError:
    print(f"LỖI: Không tìm thấy cột target '{TARGET_COLUMN}' trong dữ liệu.")
    exit()
try:
    with open(SCALER_PKL_FILE, 'rb') as f:
        scaler = pickle.load(f)
    print(f"\nĐã tải đối tượng scaler thành công từ file: {SCALER_PKL_FILE}")

except FileNotFoundError:
    print(f"\nLỖI: Không tìm thấy file SCALER '{SCALER_PKL_FILE}'. Vui lòng kiểm tra lại đường dẫn.")
    print("Bạn phải lưu đối tượng scaler từ quá trình huấn luyện để sử dụng cho dự đoán.")
    exit()
except Exception as e:
    print(f"LỖI khi tải SCALER từ file PKL: {e}")
    exit()

# --------------------------------------------------------------------------
## BƯỚC 2: TẢI MÔ HÌNH ĐÃ LƯU DƯỚI DẠNG .pkl
# --------------------------------------------------------------------------
try:
    with open(MODEL_PKL_FILE, 'rb') as file:
        model = pickle.load(file)
    
    print(f"\nĐã tải mô hình thành công từ file: {MODEL_PKL_FILE}")
    # print(f"Loại mô hình: {type(model)}") # Có thể kiểm tra loại mô hình nếu cần

except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file mô hình '{MODEL_PKL_FILE}'. Vui lòng kiểm tra lại đường dẫn.")
    exit()
except Exception as e:
    print(f"LỖI khi tải mô hình từ file PKL: {e}")
    exit()

# --------------------------------------------------------------------------
## BƯỚC 3: THỰC HIỆN DỰ ĐOÁN (PREDICTION)
# --------------------------------------------------------------------------
print("\nBắt đầu dự đoán...")

try:
    # 1. Đặt mô hình ở chế độ đánh giá (Rất quan trọng cho PyTorch)
    model.eval() 
    
    # 2. Scale dữ liệu đầu vào (Biến scaler đã được load ở bước 1.5)
    X_scaled_np = scaler.transform(X_features) 
    
    # 3. Chuyển Features đã scale thành PyTorch Tensor
    X_tensor = torch.tensor(X_scaled_np, dtype=torch.float32)
    
    # 4. Thực hiện dự đoán
    with torch.no_grad():
        # Gọi mô hình PyTorch, nó trả về y_pred và feat
        y_pred_tensor, _ = model(X_tensor) 
    
    # 5. Chuyển kết quả về mảng Numpy để lưu vào Pandas DataFrame
    predictions = y_pred_tensor.cpu().numpy().flatten()
    
    print(f"Dự đoán hoàn thành. Kích thước kết quả: {predictions.shape}")

except Exception as e:
    print(f"LỖI khi thực hiện dự đoán PyTorch: {e}")
    print("Vui lòng kiểm tra lại cấu trúc đầu vào và đầu ra của mô hình MLP.")
    exit()

# --------------------------------------------------------------------------
## BƯỚC 4: LƯU KẾT QUẢ VÀO FILE CSV MỚI
# --------------------------------------------------------------------------

# Tạo DataFrame kết quả
results_df = pd.DataFrame({
    'Y_Ground_Truth': Y_ground_truth,
    'Prediction': predictions
})

# Thêm các cột Features ban đầu vào DataFrame kết quả để dễ dàng theo dõi
# Điều này giúp bạn biết dự đoán và Ground Truth tương ứng với dữ liệu đầu vào nào
results_df = pd.concat([X, results_df], axis=1)

# Lưu DataFrame kết quả sang file CSV
results_df.to_csv(OUTPUT_CSV_FILE, index=False)

print("\n---------------------------------------------------------")
print(f"Hoàn tất! Kết quả dự đoán và Ground Truth đã được lưu vào file:")
print(f"--> {OUTPUT_CSV_FILE}")
print("---------------------------------------------------------")

print(results_df.head())