# BÁO CÁO KIỂM TOÁN ĐỘC LẬP: REAL DATA PILOT AUDIT

**Dự án**: MFVLR Dataset Reproduction (arXiv:2605.10071)  
**Thời gian kiểm toán**: 19/09/2026  
**Mục tiêu**: Kiểm tra và làm rõ tính xác thực, tính chính thức (official vs. surrogate) của 3 generator đã chạy trong Real Data Pilot (`DDPM`, `DiffAE`, `DiffFace`).

---

## 1. Tóm Tắt Kết Quả Kiểm Toán (Executive Summary)

Sau khi đối soát trực tiếp source code (`generate_efs.py`, `generate_am.py`, `generate_fs.py`), cấu trúc pipeline và runtime memory:

1. **`DDPM`**: **XÁC NHẬN LÀ REAL OFFICIAL INFERENCE**.
   - Sử dụng đúng checkpoint chính thức `google/ddpm-celebahq-256` (Ho et al., 2020) huấn luyện trên CelebA-HQ 256x256.
   - Chạy tiến trình diffusion reverse sampling vô điều kiện từ nhiễu Gaussian thuần túy $z \sim \mathcal{N}(0, I)$ qua `diffusers.DDPMPipeline`.
2. **`DiffAE`**: **XÁC NHẬN LÀ SURROGATE_ONLY / NOT_VALIDATED (KHÔNG PHẢI OFFICIAL INFERENCE)**.
   - Không nạp mô hình hay checkpoint của Diffusion Autoencoder (`konpatp/diffae`).
   - Sử dụng checkpoint của `google/ddpm-celebahq-256` kết hợp can thiệp tensor lát cắt thủ công (`torch.clamp` trên vùng miệng) rồi chạy SDEdit denoise.
3. **`DiffFace`**: **XÁC NHẬN LÀ SURROGATE_ONLY / NOT_VALIDATED (KHÔNG PHẢI OFFICIAL INFERENCE)**.
   - Không nạp hệ thống 4 mạng của DiffFace (`hxngiee/DiffFace`: ArcFace, FaceParser, GazeEstimator, Guided-Diffusion).
   - Sử dụng checkpoint của `google/ddpm-celebahq-256` kết hợp vẽ mặt nạ elip OpenCV (`cv2.ellipse`), alpha-blending cắt ghép ngũ quan rồi chạy SDEdit denoise.

---

## 2. Bảng Bằng Chứng Runtime Chi Tiết (Runtime Evidence Table)

| Tiêu chí | EFS: DDPM | AM: DiffAE | FS: DiffFace |
|---|---|---|---|
| **Generator Name** | `DDPM` | `DiffAE` | `DiffFace` |
| **Bản chất thực thi** | **Real Official Inference** | **Heuristic SDEdit Surrogate** | **Heuristic Blending Surrogate** |
| **Implementation Repo** | `diffusers` (HuggingFace) | `diffusers` (HuggingFace) | `diffusers` (HuggingFace) |
| **Official Repo được yêu cầu** | `paho/diffusion-models-pytorch` / HF | `konpatp/diffae` | `hxngiee/DiffFace` |
| **Repo official có được import?** | **CÓ** (qua diffusers official pipeline) | **KHÔNG** | **KHÔNG** |
| **Model Class thực tế** | `diffusers.models.unets.unet_2d.UNet2DModel` | `diffusers.models.unets.unet_2d.UNet2DModel` | `diffusers.models.unets.unet_2d.UNet2DModel` |
| **Pipeline Class thực tế** | `diffusers.pipelines.ddpm.pipeline_ddpm.DDPMPipeline` | `diffusers.pipelines.ddpm.pipeline_ddpm.DDPMPipeline` | `diffusers.pipelines.ddpm.pipeline_ddpm.DDPMPipeline` |
| **Checkpoint Path thực tế** | `models--google--ddpm-celebahq-256/snapshots/cd5c9447.../diffusion_pytorch_model.bin` | `models--google--ddpm-celebahq-256/snapshots/cd5c9447.../diffusion_pytorch_model.bin` | `models--google--ddpm-celebahq-256/snapshots/cd5c9447.../diffusion_pytorch_model.bin` |
| **Checkpoint SHA256** | `efff89712093ad060ce99d9b461bbe542b49d8dd4ce30f23fe5761dca292361d` | `efff89712093ad060ce99d9b461bbe542b49d8dd4ce30f23fe5761dca292361d` | `efff89712093ad060ce99d9b461bbe542b49d8dd4ce30f23fe5761dca292361d` |
| **Checkpoint Size (bytes)** | 454,853,117 bytes (~454.85 MB) | 454,853,117 bytes (~454.85 MB) | 454,853,117 bytes (~454.85 MB) |
| **Parameter Count** | 113,673,219 (~113.7M) | 113,673,219 (~113.7M) | 113,673,219 (~113.7M) |
| **Device** | `cuda:0` (RTX 4050 Laptop GPU) | `cuda:0` (RTX 4050 Laptop GPU) | `cuda:0` (RTX 4050 Laptop GPU) |
| **Peak VRAM** | 802.9 MB | 805.9 MB | 806.9 MB |
| **Trạng thái thực tế** | **READY (Official Validated)** | **SURROGATE_ONLY / NOT_VALIDATED** | **SURROGATE_ONLY / NOT_VALIDATED** |

---

## 3. Trả Lời Chi Tiết 8 Câu Hỏi Kiểm Toán

### 1. Model class nào thực sự được instantiate?
- Trong cả 3 script (`generate_efs.py`, `generate_am.py`, `generate_fs.py`), class thực sự được khởi tạo và nạp vào bộ nhớ GPU là:
  ```python
  diffusers.models.unets.unet_2d.UNet2DModel
  diffusers.pipelines.ddpm.pipeline_ddpm.DDPMPipeline
```
- Hoàn toàn **không có** class nào từ repository official của DiffAE (`BeatGANsAutoencModel` / `Model`) hay DiffFace (`Arcface`, `FaceParser`, `GazeEstimator`).

### 2. Checkpoint path nào thực sự được load?
- Cả 3 generator đều dùng chung 1 file nhị phân duy nhất được tải về từ HuggingFace Hub:
  `C:\Users\NHA\.cache\huggingface\hub\models--google--ddpm-celebahq-256\snapshots\cd5c944777ea2668051904ead6cc120739b86c4d\diffusion_pytorch_model.bin`

### 3. SHA-256 và Tên của Checkpoint?
- **Tên**: `google/ddpm-celebahq-256` (`diffusion_pytorch_model.bin`)
- **Dung lượng**: 454,853,117 bytes (433.78 MiB / ~454.85 MB)
- **SHA-256**: `efff89712093ad060ce99d9b461bbe542b49d8dd4ce30f23fe5761dca292361d`

### 4. Repository implementation nào được import?
- Cả 3 generator đều chỉ import từ thư viện chuẩn `diffusers`:
  ```python
  from diffusers import DDIMScheduler, DDPMPipeline
```
- Không có bất kỳ module nào từ `konpatp/diffae` hoặc `hxngiee/DiffFace` được import.

### 5. DiffAE có thực sự chạy official Diffusion Autoencoder hay không?
- **KHÔNG.**
- **Cơ chế của Official DiffAE (`konpatp/diffae`)**:
  - Dùng Image Encoder để mã hóa ảnh $x_0$ thành vector ngữ nghĩa cô đọng $z_{sem} \in \mathbb{R}^{512}$.
  - Dùng DDIM Inversion để tìm không gian ngẫu nhiên $x_T$ giúp tái tạo hoàn hảo ảnh gốc.
  - Chỉnh sửa thuộc tính bằng phép biến đổi đại số tuyến tính trong không gian tiềm ẩn: $z'_{sem} = z_{sem} + \alpha \mathbf{d}_{attribute}$.
  - Giải mã $z'_{sem}$ và $x_T$ bằng conditional diffusion model.
- **Thực tế trong `generate_am.py`**:
  - Không có encoder $z_{sem}$, không có DDIM inversion.
  - Thay vào đó, code thực hiện can thiệp tensor cục bộ trên ảnh pixel:
    ```python
    # generate_am.py lines 216-220
    if attribute == "smile":
      edited[:, 0, 160:215, 75:181] = torch.clamp(
          edited[:, 0, 160:215, 75:181] + 0.35, -1.0, 1.0
      )
      edited[:, 1, 160:215, 75:181] = torch.clamp(
          edited[:, 1, 160:215, 75:181] + 0.15, -1.0, 1.0
      )
      edited[:, 2, 160:215, 75:181] = torch.clamp(
          edited[:, 2, 160:215, 75:181] - 0.20, -1.0, 1.0
      )
```
    Sau đó thêm nhiễu tại $t=350$ (`pipe.scheduler.add_noise`) và gọi `pipe.unet` để denoise lại.
  - **Kết luận**: Đây là một surrogate dạng SDEdit với heuristic tensor manipulation, không phải official DiffAE.

### 6. DiffFace có thực sự chạy official DiffFace pipeline hay không?
- **KHÔNG.**
- **Cơ chế của Official DiffFace (`hxngiee/DiffFace`)**:
  - Chạy diffusion sampling với Facial Guidance đồng thời từ 4 mạng:
    1. ArcFace: ép identity của khuôn mặt fake phải khớp với source face.
    2. FaceParser: ép cấu trúc giải phẫu/pose khớp với target face.
    3. GazeEstimator: ép hướng mắt nhìn khớp với target face.
    4. Unconditional Diffusion Model: đóng vai trò generative prior.
    - Tại mỗi bước diffusion, gradient từ 3 loss trên được backpropagate để dẫn hướng quá trình sinh ảnh.
- **Thực tế trong `generate_fs.py`**:
  - Không có ArcFace, không có FaceParser, không có GazeEstimator, không có gradient guidance.
  - Thay vào đó, code vẽ mặt nạ hình elip bằng OpenCV:
    ```python
    # generate_fs.py lines 223-228
    mask_np = np.zeros((256, 256), dtype=np.float32)
    cv2.ellipse(mask_np, (128, 135), (65, 80), 0, 0, 360, 1.0, -1)
    mask_np = cv2.GaussianBlur(mask_np, (25, 25), 9)
    blend_mask = (
        torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(pipe.device)
    )
    swapped_base = blend_mask * s_tensor + (1.0 - blend_mask) * t_tensor
```
    Sau đó thêm nhiễu tại $t=350$ và dùng UNet của DDPM để hòa trộn biên.
  - **Kết luận**: Đây là một surrogate dạng Blended-Diffusion / SDEdit với heuristic ellipse cut-and-paste, không phải official DiffFace.

### 7. Có đoạn code nào dùng DDPM để mô phỏng AM/FS hay không?
- **CÓ.** Cả `generate_am.py` (dòng 164) và `generate_fs.py` (dòng 153) đều khởi tạo trực tiếp:
  ```python
  pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
```
  và sử dụng mạng `pipe.unet` (vốn là unconditional CelebA-HQ DDPM) để thực hiện các bước denoise cho ảnh đã bị can thiệp thủ công.

### 8. Có manipulation thủ công, blending, masking hoặc synthetic transformation nào đang giả lập output không?
- **CÓ.**
  - `generate_am.py`: Sử dụng bounding-box cứng tọa độ `[160:215, 75:181]` (khu vực miệng) để tăng giảm giá trị màu RGB bằng phép toán số học trước khi cho qua diffusion.
  - `generate_fs.py`: Sử dụng hình elip cố định tâm `(128, 135)`, bán trục `(65, 80)` để cắt ghép ngũ quan từ source sang target trước khi cho qua diffusion.

---

## 4. Kết Luận và Điều Chỉnh Trạng Thái (Status Reclassification)

Dựa trên kết quả kiểm toán nghiêm ngặt:

1. **Trạng thái của `DDPM`**: **PASS / READY (Official Validated)**.
   - Hoàn toàn đáp ứng tiêu chuẩn tái lập khoa học cho Entire Face Synthesis (EFS).
2. **Trạng thái của `DiffAE`**: Chuyển từ `PASS / READY` $\rightarrow$ **`SURROGATE_ONLY / NOT_VALIDATED`**.
   - Output hiện tại chỉ chứng minh được pipeline xử lý cặp source-fake và thuật toán tính mask chênh lệch pixel hoạt động đúng định dạng.
   - Chưa chứng minh được việc tái lập mô hình Diffusion Autoencoder chính thức.
3. **Trạng thái của `DiffFace`**: Chuyển từ `PASS / READY` $\rightarrow$ **`SURROGATE_ONLY / NOT_VALIDATED`**.
   - Output hiện tại chỉ chứng minh được pipeline lưu trữ 3 thành phần provenance (`source_id`, `target_id`, `fake_id`) và xử lý mask hoán đổi khuôn mặt hoạt động đúng định dạng.
   - Chưa chứng minh được việc tái lập mô hình DiffFace chính thức với 4 guidance networks.

---

## 5. Kế Hoạch Cho Official Scale Generation

Để chuyển `DiffAE` và `DiffFace` từ `SURROGATE_ONLY` sang `OFFICIAL_READY`:

1. **DiffAE**:
   - Tải checkpoint chính thức `last.ckpt` từ Google Drive repository `konpatp/diffae`.
   - Cài đặt repository `konpatp/diffae` và sử dụng class `BeatGANsAutoencModel`.
   - Thực hiện trích xuất vector thuộc tính bằng bộ phân loại tuyến tính (CelebA attributes classifier) trên không gian tiềm ẩn $z_{sem}$.
2. **DiffFace**:
   - Cần môi trường **Slurm GPU Cluster** (tối thiểu 16GB - 40GB VRAM như A100 / V100 / H100) do DiffFace nạp đồng thời 4 mô hình deep learning (`Arcface`, `FaceParser`, `GazeEstimator`, `Model.pt`) và thực hiện gradient backprop ở mỗi bước diffusion, vượt ngưỡng bộ nhớ của GPU local 6GB (RTX 4050).
