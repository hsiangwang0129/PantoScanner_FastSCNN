from fast_scnn.fast_scnn import FastSCNN  # 這是 repo 裡定義的 FastSCNN class
import torch
# 1. 先還原模型結構
model = FastSCNN(num_classes=2)  # 根據當初訓練用的設定

# 2. 載入已經訓練好的 checkpoint
checkpoint = torch.load("/Users/shawn/Desktop/computer_vision/PantoScanner_WEPSnet/app/segmentation_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 這邊依照你的模型輸入尺寸來建假資料
example_input = torch.randn(1, 3, 3200, 315)  # 依據訓練時 input shape 修改

# trace
traced_model = torch.jit.trace(model, example_input)

# 儲存成 .pt
traced_model.save("fastscnn_traced.pt")


