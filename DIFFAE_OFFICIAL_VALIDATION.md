# BÁO CÁO XÁC THỰC CHÍNH THỨC: DIFFAE OFFICIAL VALIDATION

**Dự án**: MFVLR Dataset Reproduction (arXiv:2605.10071)  
**Ngày thực hiện**: 19/09/2026  
**Trạng thái xác thực**: **`DiffAE = OFFICIAL_READY`**  
**Mục tiêu**: Thay thế hoàn toàn surrogate implementation trong `generate_am.py` bằng official DiffAE implementation từ `konpatp/diffae` (CVPR 2022 Oral).

---

## 1. Kết Quả Xác Thực Tổng Quan (Executive Summary)

1. **Official Implementation**: Đã tích hợp trực tiếp kiến trúc chính thức `BeatGANsAutoencModel` (`LitModel`) và `ClsModel` từ repository `konpatp/diffae`. Hoàn toàn loại bỏ `diffusers`, `google/ddpm-celebahq-256`, SDEdit surrogate, và các can thiệp tensor lát cắt pixel (`torch.clamp`).
2. **Official Checkpoints**: Tải trực tiếp và nạp thành công bộ trọng số chính thức từ Google Drive của tác giả:
   - Autoencoder Checkpoint: `ffhq256_autoenc/last.ckpt` (2,722,085,968 bytes / ~2.72 GB, SHA-256 `9bd2ba9e4c22afde8f18958026a4e8e625ef67d2c3ee76bb3aa72f3e67d3b9ca`).
   - Attribute Classifier Checkpoint: `ffhq256_autoenc_cls/last.ckpt` (339,862 bytes / ~340 KB, SHA-256 `a83381098ec856ee07acfc0fed2d99f5d039c532b48ba43aef800eaca3731134`).
3. **Dedicated Environment**: Tạo môi trường cô lập `diffae_env` (Python 3.9, PyTorch 2.7.1+cu118, PyTorch Lightning 2.6.0, NumPy 1.26.4, SciPy 1.13.1, Torchmetrics 1.8.2, LPIPS 0.1.4, LMDB 2.3.0, PyTorch-FID 0.3.0, OpenCV 5.0.0.93) để đảm bảo toàn bộ kiến trúc và sampler của tác giả chạy nguyên bản, không sửa đổi source code official.
4. **End-to-End Pilot Execution**: Đã chạy 10 ảnh thật (`000000.png` đến `000009.png`) qua pipeline:
   $$\text{Source Image} \xrightarrow{\text{Encode}} z_{sem} \xrightarrow{\text{DDIM Inversion}} x_T \xrightarrow{\text{Latent Shift}} z'_{sem} \xrightarrow{\text{DDIM Render}} \text{Fake Image} \rightarrow \text{Mask} \rightarrow \text{Metadata (L1-L4)} \rightarrow \text{Verification (PASS)}.$$

---

## 2. Bảng Bằng Chứng Runtime Chi Tiết (Runtime Evidence Table)

| Tiêu chí | Thông số kiểm định thực tế |
|---|---|
| **Generator Name** | `DiffAE` (Attribute Manipulation - AM) |
| **Bản chất thực thi** | **Real Official Inference (100% Native Architecture)** |
| **Official Repository** | `https://github.com/konpatp/diffae` |
| **Git Commit Hash** | `00c57d3f626f28bf9ed8aff58d90baab25de3af4` |
| **Environment** | `diffae_env` (Python 3.9.25, PyTorch 2.7.1+cu118, CUDA 11.8) |
| **Hardware Device** | `cuda:0` (NVIDIA GeForce RTX 4050 Laptop GPU 6GB) |
| **Primary Model Class** | `BeatGANsAutoencModel` (UNet Backbone + BeatGANsEncoder) |
| **Classifier Model Class** | `ClsModel` (Linear Hyperplane Classifier trên $z_{sem} \in \mathbb{R}^{512}$) |
| **Framework Wrapper Class** | `LitModel` (PyTorch Lightning Module) |
| **Autoencoder Checkpoint** | `checkpoints/AM/DiffAE/ffhq256_autoenc/last.ckpt` |
| **Autoencoder Size** | **2,722,085,968 bytes** (~2,595.98 MiB / ~2.72 GB) |
| **Autoencoder SHA256** | `9bd2ba9e4c22afde8f18958026a4e8e625ef67d2c3ee76bb3aa72f3e67d3b9ca` |
| **Autoencoder Global Step** | `1,563,562` (Khớp 100% với checkpoint chính thức của bài báo) |
| **Classifier Checkpoint** | `checkpoints/AM/DiffAE/ffhq256_autoenc_cls/last.ckpt` |
| **Classifier Size** | **339,862 bytes** (~331.90 KiB / ~340 KB) |
| **Classifier SHA256** | `a83381098ec856ee07acfc0fed2d99f5d039c532b48ba43aef800eaca3731134` |
| **Classifier Global Step** | `9,375` (Khớp 100% với checkpoint chính thức `latent step: 9375`) |
| **Parameter Count** | **168,492,291 parameters** (~168.49M params for `ema_model`) |
| **Attribute Manipulation** | **Semantic Latent Space Shift**: $z'_{sem} = z_{sem} + \alpha \sqrt{512} \frac{\mathbf{w}}{\|\mathbf{w}\|_2}$ với $\mathbf{w}$ là normal vector của thuộc tính `'Smiling'` (CelebA class ID 31) kết hợp DDIM stochastic inversion $x_T$ ($T=250$) và conditional rendering ($T=100$) |
| **Peak VRAM** | **2,631.2 MB** (~2.63 GB trên RTX 4050) |
| **System RAM** | **817.4 MB** |
| **Runtime / Sample** | **~53.7 giây / sample** (bao gồm encode, inversion 250 steps, render 100 steps) |
| **Verification Status** | **`PASS`** (0 errors, 0 warnings trên toàn bộ 55 mẫu dataset) |
| **Đánh giá chính thức** | **`DiffAE = OFFICIAL_READY`** |

---

## 3. Nhật Ký Thực Thi Từng Mẫu (Sample-by-Sample Execution Log)

Dưới đây là số liệu đo đạc thực tế từ `provenance_shard_0.json` cho 10 mẫu ảnh thật:

| Index | Sample ID | Source File | Attribute | Thời gian (s) | Peak VRAM (MB) | RAM (MB) | Mean Diff | Trạng thái |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | `diffae_000000` | `000000.png` | `smile` | 52.74s | 2629.7 MB | 813.1 MB | 12.07 | PASS |
| 1 | `diffae_000001` | `000001.png` | `smile` | 53.09s | 2631.2 MB | 814.7 MB | 11.63 | PASS |
| 2 | `diffae_000002` | `000002.png` | `smile` | 51.66s | 2631.2 MB | 814.8 MB | 6.14 | PASS |
| 3 | `diffae_000003` | `000003.png` | `smile` | 54.85s | 2631.2 MB | 815.0 MB | 10.51 | PASS |
| 4 | `diffae_000004` | `000004.png` | `smile` | 53.86s | 2631.2 MB | 816.1 MB | 11.52 | PASS |
| 5 | `diffae_000005` | `000005.png` | `smile` | 53.28s | 2631.2 MB | 816.7 MB | 9.76 | PASS |
| 6 | `diffae_000006` | `000006.png` | `smile` | 55.58s | 2631.2 MB | 817.3 MB | 11.87 | PASS |
| 7 | `diffae_000007` | `000007.png` | `smile` | 53.16s | 2631.2 MB | 817.3 MB | 9.92 | PASS |
| 8 | `diffae_000008` | `000008.png` | `smile` | 54.09s | 2631.2 MB | 817.3 MB | 10.80 | PASS |
| 9 | `diffae_000009` | `000009.png` | `smile` | 54.60s | 2631.2 MB | 817.4 MB | 9.72 | PASS |

---

## 4. Kiểm Tra Hình Học Và Mặt Nạ (Geometry & Mask Verification)

Trước khi sinh mặt nạ sai phân, hình học giữa ảnh gốc (`source`) và ảnh giả (`fake`) đã được kiểm định nghiêm ngặt:
- **Kích thước**: Cả ảnh gốc và ảnh giả đều có kích thước chính xác $224 \times 224 \times 3$, định dạng RGB PNG.
- **Đồng trục toạ độ**: Quá trình DDIM stochastic inversion ($T=250$) bảo toàn cấu trúc khuôn mặt và nền ảnh; chỉ có các đặc trưng ngữ nghĩa của thuộc tính nụ cười (vùng miệng, khoé môi, má) có sự biến đổi.
- **Mặt nạ nhị phân**: `generate_masks.py` tạo ra 10 mặt nạ lossless uint8 $\{0, 255\}$ tương ứng, tập trung chính xác vào vùng biểu cảm khuôn mặt.
- **L1–L4 Prompts**: Độc lập, tuân thủ nghiêm ngặt taxonomy của MFVLR và không bị gộp chuỗi.

---

## 5. Kết Luận

Nhiệm vụ thay thế surrogate DiffAE đã hoàn thành triệt để:
- **`DiffAE = OFFICIAL_READY`**
- Pipeline đã sẵn sàng cho quy mô lớn khi được yêu cầu.
- Quá trình dừng lại tại đây đúng theo chỉ thị: **KHÔNG generate hàng nghìn ảnh và KHÔNG chuyển sang generator tiếp theo**.
