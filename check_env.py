import torch
import snntorch
import rasterio
import geopandas

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device="cuda")
    print("GPU test:", (x @ x).sum().item() != 0)
print("snnTorch:", snntorch.__version__)
print("rasterio:", rasterio.__version__, "| GDAL:", rasterio.__gdal_version__)
print("geopandas:", geopandas.__version__)
