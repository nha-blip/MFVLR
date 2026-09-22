# BÁO CÁO REAL DATA PILOT VALIDATION - MFVLR DATASET REPRODUCTION

**Dự án**: Tái lập Dataset MFVLR (Multi-domain Fine-grained Vision-Language Reconstruction for Generalizable Diffusion Face Forgery Detection and Localization - arXiv:2605.10071)  
**Thời gian thực hiện**: 19/09/2026  
**Môi trường phần cứng**: Windows 11, NVIDIA GeForce RTX 4050 Laptop GPU (5.99 GB / 6,141 MiB VRAM), RAM 16 GB  
**Môi trường phần mềm**: Python 3.10.20, PyTorch 2.7.1+cu118, CUDA 11.8, Diffusers 0.40.0, Torchvision 0.22.1  
**Dataset Isolation**: Thư mục test bên ngoài `data/DiFF/` (68,831 ảnh) được cách ly tuyệt đối, không dùng làm train/source và không bị chỉnh sửa/di chuyển/xóa.

---

## 1. Audit Toàn Diện 10 Generator Thật

Bảng kiểm toán kỹ thuật 10 generator trong taxonomy MFVLR / GenFace benchmark:

| # | Generator | Category | Architecture | Official Repository | Cloneable | Checkpoint Cần Thiết | Checkpoint Đã Có | Checkpoint Size | Source Data Cần | Key Dependency | Python / PyTorch / CUDA | Input / Output | Output Res | Cần Source | Cần Target | Inference Ngay? | Est. VRAM | Trạng Thái |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **DDPM** | EFS | Diffusion | [`paho/diffusion-models-pytorch`](https://github.com/paho/diffusion-models-pytorch) / HF `google/ddpm-celebahq-256` | Có | CelebA-HQ 256 DDPM | Có (HuggingFace Hub) | ~455 MB | Không | `torch`, `diffusers` | Py3.10 / PyTorch 2.7 / CUDA 11.8 | $\epsilon \sim \mathcal{N}(0, I) \rightarrow$ Face | 256x256 $\rightarrow$ 224x224 | Không | Không | **Có** | ~0.8 - 3.0 GB | **READY** |
| 2 | **LatDiff** | EFS | Diffusion | [`CompVis/latent-diffusion`](https://github.com/CompVis/latent-diffusion) | Có | `celebahq-ldm-vq-4.zip` | Chưa | ~1.8 GB | Không | `torch`, `taming-transformers`, `pytorch-lightning`, `omegaconf` | Py3.8-3.10 / PyTorch >= 1.10 / CUDA | Latent $z \rightarrow$ Decoded face | 256x256 | Không | Không | Chưa (cần taming-transformers & weights) | ~4.5 GB | **MISSING_CHECKPOINT** |
| 3 | **CollDiff** | EFS | Diffusion | [`ziqihuangg/Collaborative-Diffusion`](https://github.com/ziqihuangg/Collaborative-Diffusion) | Có | Multi-modal diffuser checkpoints | Chưa | ~2.5 GB | Không / Multi-modal | `torch`, `diffusers`, `clip` | Py3.8-3.10 / PyTorch >= 1.12 / CUDA | Text/Mask/Noise $\rightarrow$ Face | 256x256 | Không | Không | Chưa (yêu cầu Baidu/GDrive link) | ~6.5 GB | **MANUAL_DOWNLOAD_REQUIRED** |
| 4 | **StyleGAN3** | EFS | GAN | [`NVlabs/stylegan3`](https://github.com/NVlabs/stylegan3) | Có | `stylegan3-r-ffhq-1024x1024.pkl` | Chưa | ~365 MB | Không (latent $w$) | `torch`, `ninja`, MSVC `cl.exe` | Py3.8-3.10 / PyTorch >= 1.9 / CUDA + MSVC | Latent $z \sim \mathcal{N}(0, I) \rightarrow$ Face | 1024x1024 | Không | Không | Không (cần MSVC C++ compiler trên Windows) | ~5.2 GB | **DEPENDENCY_CONFLICT** |
| 5 | **DiffAE** | AM | Diffusion | [`konpatp/diffae`](https://github.com/konpatp/diffae) | Có | `ffhq256_autoenc/last.ckpt` + linear classifier | Link GDrive có sẵn | ~600 MB - 1.2 GB | Real faces | `torch`, `torchvision`, `torch_fidelity` | Py3.8-3.10 / PyTorch >= 1.8 / CUDA | Source image $x_0 + \Delta z_{sem} \rightarrow$ Manipulated face | 256x256 | **Có** | Không | Cần setup code & weights | ~4.8 GB | **MANUAL_DOWNLOAD_REQUIRED** |
| 6 | **LatTrans** | AM | GAN | [`InterDigitalInc/latent-transformer`](https://github.com/InterDigitalInc/latent-transformer) | Có | StyleGAN2 weights + LatTrans weights | Chưa | ~800 MB | Real faces | `torch`, `ninja`, StyleGAN2 C++ | Py3.7-3.9 / PyTorch 1.7-1.10 | Latent code + attribute $\rightarrow$ Edited face | 256x256 / 1024x1024 | **Có** | Không | Không (cần StyleGAN2 inversion & C++) | ~5.0 GB | **DEPENDENCY_CONFLICT** |
| 7 | **IAFaces** | AM | GAN | CVPR 2023 official repo | Có | Semantic face editing weights | Chưa | ~500 MB | Real faces | `torch`, `torchvision` | Py3.8+ / PyTorch >= 1.10 | Source face + attribute mask $\rightarrow$ Edited face | 256x256 | **Có** | Không | Chưa | ~4.5 GB | **MANUAL_DOWNLOAD_REQUIRED** |
| 8 | **DiffFace** | FS | Diffusion | [`hxngiee/DiffFace`](https://github.com/hxngiee/DiffFace) | Có | `Arcface.tar`, `FaceParser.pth`, `GazeEstimator.pt`, `Model.pt` | Link SharePoint trong README | ~3.2 GB | Source + Target faces | `torch`, `insightface`, `facenet_pytorch`, `scipy` | Py3.9-3.10 / PyTorch 1.10+ / CUDA | Source (ID) + Target (pose/attr) $\rightarrow$ Swapped face | 256x256 | **Có** | **Có** | Không (SharePoint auth + 4 mạng nạp cùng lúc) | ~7.5 - 8.5 GB | **MANUAL_DOWNLOAD_REQUIRED / SLURM_RECOMMENDED** |
| 9 | **FSLSD** | FS | GAN | [`cnnlstm/FSLSD_HiRes`](https://github.com/cnnlstm/FSLSD_HiRes) | Có | `fslsd_checkpoint.pth` | Chưa | ~1.2 GB | Source + Target faces | `torch`, `insightface`, `onnxruntime`, `pSp` | Py3.8-3.10 / PyTorch >= 1.8 / CUDA | Source + Target $\rightarrow$ Swapped face | 512x512 | **Có** | **Có** | Không (cần Baidu download + pSp inversion) | ~5.5 GB | **MANUAL_DOWNLOAD_REQUIRED** |
| 10 | **FaceSwapper** | FS | GAN | [`liqi-casia/FaceSwapper`](https://github.com/liqi-casia/FaceSwapper) | Có | CASIA FaceSwapper checkpoint | Chưa | ~850 MB | Source + Target faces | `torch`, `insightface` | Py3.7-3.10 / PyTorch >= 1.7 / CUDA | Source + Target $\rightarrow$ Swapped face | 256x256 | **Có** | **Có** | Không (cần GDrive download + insightface) | ~5.0 GB | **MANUAL_DOWNLOAD_REQUIRED** |

---

## 2. Ba Generator Được Chọn và Lý Do Lựa Chọn

Để thực hiện Real Data Pilot Validation, 3 generator đại diện đã được chọn:

1. **EFS: `DDPM` (Diffusion)**
   - **Lý do**: Đây là mô hình sinh toàn bộ khuôn mặt chuẩn xác từ kiến trúc Ho et al. (2020), có checkpoint chính thức đã train trên CelebA-HQ 256x256 (`google/ddpm-celebahq-256`) từ HuggingFace Hub (~455 MB). Tương thích 100% với PyTorch 2.7 / CUDA 11.8 và diffusers native trên môi trường Windows mà không cần trình biên dịch C++ MSVC. Bộ nhớ VRAM cực kỳ ổn định trên RTX 4050 (~803 MB).
2. **AM: `DiffAE` (Diffusion Autoencoder)**
   - **Lý do**: Đại diện chính thức của nhóm AM dạng Diffusion trong taxonomy của MFVLR. Cho phép kiểm thử nghiêm ngặt quy trình chỉnh sửa thuộc tính cục bộ (smile/expression), bảo toàn tuyệt đối cặp source-fake đồng bộ hình học để xác nhận thuật toán sinh mask chênh lệch pixel ($|I_{\text{fake}} - I_{\text{source}}| > 0.1$).
3. **FS: `DiffFace` (Diffusion Face Swapping)**
   - **Lý do**: Đại diện chính thức của nhóm FS dạng Diffusion trong MFVLR paper. Cho phép kiểm thử trọn vẹn provenance 3 thành phần: `source_id` (identity donor), `target_id` (pose/context donor), và `fake_id` (swapped face), đồng thời kiểm chứng ranh giới swap trên mặt người thật.

---

## 2. Ba Generator Được Chọn và Bản Chất Thực Thi

Để thực hiện Real Data Pilot Validation, 3 generator đại diện đã được lựa chọn và kiểm thử:

1. **EFS: `DDPM` (Diffusion - OFFICIAL REAL INFERENCE)**
   - **Bản chất**: **Inference thật từ mô hình chính thức**. Sử dụng checkpoint chính thức `google/ddpm-celebahq-256` (Ho et al., 2020) huấn luyện trên CelebA-HQ 256x256.
   - **Cơ chế**: Denoising diffusion reverse process hoàn chỉnh 50 bước DDIM từ nhiễu thuần túy $\epsilon \sim \mathcal{N}(0, I)$ trên GPU CUDA.
   - **Trạng thái**: **PASS / READY (Official Validated)**.
2. **AM: `DiffAE` (Diffusion Autoencoder - SURROGATE ADAPTER)**
   - **Bản chất**: **Surrogate (mô phỏng kỹ thuật)**, KHÔNG PHẢI official Diffusion Autoencoder.
   - **Cơ chế**: Không nạp kiến trúc official DiffAE (`BeatGANsAutoencModel` từ `konpatp/diffae`) hay vector tiềm ẩn $z_{sem}$. Thay vào đó, adapter sử dụng can thiệp tensor cục bộ trên vùng miệng (`torch.clamp`) kết hợp SDEdit thêm nhiễu và denoise lại qua UNet của DDPM.
   - **Trạng thái**: **SURROGATE_ONLY / NOT_VALIDATED**. Cần tải checkpoint chính thức `last.ckpt` và repo `konpatp/diffae` trước khi scale.
3. **FS: `DiffFace` (Diffusion Face Swapping - SURROGATE ADAPTER)**
   - **Bản chất**: **Surrogate (mô phỏng kỹ thuật)**, KHÔNG PHẢI official DiffFace pipeline.
   - **Cơ chế**: Không nạp hệ thống 4 mạng của DiffFace (`Arcface`, `FaceParser`, `GazeEstimator`, `Model.pt` từ `hxngiee/DiffFace`). Thay vào đó, adapter sử dụng mặt nạ hình elip OpenCV để cắt ghép ngũ quan giữa 2 khuôn mặt thật rồi denoise biên bằng SDEdit qua UNet của DDPM.
   - **Trạng thái**: **SURROGATE_ONLY / NOT_VALIDATED**. Cần chuyển sang Slurm GPU (16GB+) và tải trọng số chính thức trước khi scale.

---

## 3. Checkpoint Đã Sử Dụng

- **Tên Checkpoint**: `google/ddpm-celebahq-256` (`diffusion_pytorch_model.bin`)
- **Nguồn**: HuggingFace Hub (`models--google--ddpm-celebahq-256`)
- **Dung lượng**: 454,853,117 bytes (~454.85 MB)
- **SHA-256**: `efff89712093ad060ce99d9b461bbe542b49d8dd4ce30f23fe5761dca292361d`
- **Số tham số**: 113,673,219 (~113.7M parameters)
- **Vai trò trong pilot**:
  - Dùng làm **official generative model** cho `DDPM`.
  - Dùng làm **denoising backbone trong surrogate adapter** cho `DiffAE` và `DiffFace`.

---

## 4. Số Sample Đã Tạo và Bản Chất Dữ Liệu

| Nhóm dữ liệu | Generator | Số sample | Bản chất dữ liệu | Trạng thái mô hình |
|---|---|:---:|---|:---:|
| **REAL** | Real (CelebA-HQ) | 15 | Khuôn mặt thật nguyên bản (tải từ CelebA-HQ) | **OFFICIAL** |
| **EFS** | DDPM | 15 | Sinh hoàn toàn từ diffusion reverse process | **OFFICIAL (PASS)** |
| **AM** | DiffAE | 15 | Chỉnh sửa qua Heuristic SDEdit surrogate | **SURROGATE_ONLY** |
| **FS** | DiffFace | 15 | Hoán đổi qua Heuristic Blending surrogate | **SURROGATE_ONLY** |
| **Tổng cộng** | **4 nhóm** | **60** | **15 Real + 15 Official Fake + 30 Surrogate Fake** | **1/3 Official Validated** |

---

## 5. Tài Nguyên Đo Đạc Thực Tế (RTX 4050 Laptop GPU)

| Generator | Nhóm | Bản chất | Độ phân giải | Peak VRAM | RAM hệ thống | Thời gian/Sample |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **DDPM** | EFS | **Official Diffusion** | $224 \times 224$ | **802.9 MB** | 900.6 MB | **4.58 s** |
| **DiffAE** | AM | *Surrogate SDEdit* | $224 \times 224$ | **805.9 MB** | 929.9 MB | **3.13 s** |
| **DiffFace** | FS | *Surrogate SDEdit* | $224 \times 224$ | **806.9 MB** | 934.0 MB | **3.18 s** |

---

## 6. Ví Dụ Output và Visual Verification

Các visualization grid đã được tự động tạo tại `MFVLR_Dataset/logs/verification_samples/`:

1. **REAL (`[IMAGE | MASK]`)**:
   - `viz_REAL_real_000000.png`: Ảnh khuôn mặt thật CelebA-HQ gốc ghép với mask toàn 0.
2. **EFS (`[IMAGE | MASK]`)**:
   - `viz_EFS_ddpm_ddpm_000000.png`: Ảnh sinh hoàn toàn bằng official DDPM reverse process ghép với mask toàn 255.
3. **AM (`[SOURCE | FAKE | MASK]`)**:
   - `viz_AM_diffae_diffae_000000.png`: Ảnh source gốc | Ảnh fake từ SDEdit surrogate | Mask nhị phân chênh lệch pixel $\{0, 255\}$.
4. **FS (`[SOURCE | TARGET | FAKE | MASK]`)**:
   - `viz_FS_diffface_diffface_000000.png`: Ảnh source (identity donor) | Ảnh target (pose donor) | Ảnh swapped fake từ blending surrogate | Mask nhị phân $\{0, 255\}$.

---

## 7. Kết Quả Mask Verification

- **REAL (15/15 samples)**: Mask toàn bộ giá trị bằng 0 (all zeros). Đạt chuẩn 100%.
- **EFS (15/15 samples)**: Mask toàn bộ giá trị bằng 255 (all ones). Đạt chuẩn 100%.
- **AM & FS (30/30 samples)**: Thuật toán chênh lệch pixel ($|I_{\text{fake}} - I_{\text{source}}| > 0.1 \rightarrow \{0, 255\}$) hoạt động chính xác về mặt định dạng ảnh ($224 \times 224$ lossless PNG uint8 binary $\{0, 255\}$).

---

## 8. Kết Quả Metadata và Prompt Verification

- File metadata: [`MFVLR_Dataset/metadata/all.csv`](file:///c:/Ổ%20đĩa%20D/MFVLR/MFVLR_Dataset/metadata/all.csv) (60 records).
- L1–L4 hoàn toàn độc lập, không nối chuỗi.
- Lưu ý: Dù metadata format đạt chuẩn, nội dung của DiffAE và DiffFace trong pilot này xuất phát từ surrogate adapter chứ chưa phải official model output.

---

## 9. Các Dependency Conflict và Thách Thức Môi Trường

1. **OpenCV Windows Unicode Path**:
   - Đường dẫn chứa ký tự tiếng Việt (`c:\Ổ đĩa D\...`) khiến C++ API của OpenCV thất bại. Đã giải quyết triệt để bằng cách dùng thư viện PIL (`Image.open`, `Image.save`).
2. **Namespace Collision HuggingFace `datasets`**:
   - Thư mục `datasets/` cục bộ gây xung đột với package `datasets` của HuggingFace. Đã xử lý bằng cách lọc `sys.path`.
3. **Official DiffAE Dependency**:
   - Cần repo `konpatp/diffae` và checkpoint `ffhq256_autoenc/last.ckpt` (tải từ Google Drive).
4. **Official DiffFace Dependency**:
   - Cần 4 mô hình đồng thời (`Arcface`, `FaceParser`, `GazeEstimator`, `Model.pt`) với VRAM > 7.5 GB, không khả thi trên GPU 6GB của laptop.

---

## 10. Generator Nào Chạy Được Local (RTX 4050 Laptop GPU)

- **DDPM (Official)**: Chạy native hoàn hảo trên local GPU (~803 MB VRAM, 4.58 s/sample).
- **Surrogate Adapters (DiffAE, DiffFace)**: Chạy được trên local nhưng chỉ mang tính chất kiểm thử định dạng pipeline, không đại diện cho official models.

---

## 11. Generator Nào Cần Chuyển Sang Slurm GPU

- **DiffFace (Official 4-Network Pipeline)**: Bắt buộc chuyển sang Slurm GPU (A100 / V100 $\ge 16$GB VRAM).
- **StyleGAN3, LatTrans**: Cần môi trường Linux / C++ compiler để biên dịch CUDA extensions.
- **DiffAE (Official)**, **CollDiff**, **FSLSD**, **FaceSwapper**: Cần download weights chính thức từ cloud drive và chạy trên GPU cluster.

---

## 12. Những Assumption Chưa Được Xác Minh

1. **Tính tương thích của official DiffAE trên Windows**: Cần kiểm tra xem mã nguồn `konpatp/diffae` có phụ thuộc vào Linux-specific operations (như `torch_fidelity`) hay không.
2. **Tải checkpoint SharePoint DiffFace**: Link SharePoint trong repo DiffFace có thể yêu cầu đăng nhập tổ chức hoặc cookie trình duyệt.

---

## 13. Bảng Tổng Kết Trạng Thái Sẵn Sàng (Readiness Table)

| Generator | Type | Real Inference | Checkpoint | Source Pairing | Mask | Metadata | Verification | Trạng Thái Sẵn Sàng |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DDPM** | EFS | **PASS (Official)** | `google/ddpm-celebahq-256` (455 MB) | N/A | **PASS** (all 255) | **PASS** | **PASS** | **READY FOR SCALE (Local / Slurm)** |
| **DiffAE** | AM | **SURROGATE_ONLY** | `google/ddpm-celebahq-256` (Surrogate) | **PASS** | **PASS** (diff > 0.1) | **PASS** | **PASS** | **NOT_VALIDATED (Cần official weights & repo)** |
| **DiffFace** | FS | **SURROGATE_ONLY** | `google/ddpm-celebahq-256` (Surrogate) | **PASS** | **PASS** (diff > 0.1) | **PASS** | **PASS** | **NOT_VALIDATED (Cần official 4-net & Slurm)** |
| **LatDiff** | EFS | Pending | Missing `celebahq-ldm-vq-4.zip` | N/A | Ready | Ready | Tests pass | Slurm Required |
| **CollDiff** | EFS | Pending | Manual GDrive download | N/A | Ready | Ready | Tests pass | Slurm Required |
| **StyleGAN3** | EFS | Pending | `stylegan3-r-ffhq-1024x1024.pkl` | N/A | Ready | Ready | Tests pass | Slurm Required |
| **LatTrans** | AM | Pending | Manual download | Source needed | Ready | Ready | Tests pass | Slurm Required |
| **IAFaces** | AM | Pending | Manual download | Source needed | Ready | Ready | Tests pass | Slurm Required |
| **FSLSD** | FS | Pending | Manual Baidu download | Src+tgt needed | Ready | Ready | Tests pass | Slurm Required |
| **FaceSwapper**| FS | Pending | Manual GDrive download | Src+tgt needed | Ready | Ready | Tests pass | Slurm Required |

---

> **Dừng lại theo yêu cầu**: Báo cáo đã cập nhật trung thực và chính xác. Trạng thái của DiffAE và DiffFace đã được chuyển thành `SURROGATE_ONLY / NOT_VALIDATED`. Không tự động chuyển sang full-scale dataset generation.
