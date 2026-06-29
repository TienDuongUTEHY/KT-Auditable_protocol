# Science Paper P0 — Đồ thị đa quan hệ & Knowledge Tracing

Mã nguồn hỗ trợ luận chứng và thí nghiệm cho chủ đề **xây dựng đồ thị kỹ năng đa quan hệ có kiểm soát rò rỉ dữ liệu (leakage)**, phục vụ **theo dõi tri thức (Knowledge Tracing)** khi các kỹ năng (KC) có mức độ **thưa (sparse)**.

Tên cấu hình nội bộ trong `configs/global.yaml`: `KT-Auditable_protocol`; tiêu đề đặt trong cùng file: **"Leakage-Controlled Multi-Relational Skill Graph Construction for Sparse-Skill Knowledge Tracing"**.

---

## Mục tiêu dự án

- **Tiền xử lý và chia dữ liệu** theo học sinh và thứ tự thời gian (`learner_based_temporal`), với các tỷ lệ train/valid/test cấu hình trong `configs/global.yaml`.
- **Ba loại quan hệ giữa kỹ năng** chỉ được ước lượng từ **tập huấn luyện** (giảm rò rỉ):
  - **E_pre** — tiền đề / thứ tự dạy học gợi ý,
  - **E_sim** — tương đồng (ví dụ Jaccard trên `question_id` chung),
  - **E_co** — đồng xuất hiện (ví dụ PMI có làm trơn, chỉ cạnh dương).
- **Kiểm toán và báo cáo**: nguồn gốc Q-matrix, DAG, báo cáo “eco”, **kiểm toán leakage (L1–L5)**, thống kê đồ thị, hồ sơ “sparse skill”, baseline có **ablation** theo biến thể đồ thị (`no_graph`, `E_pre`, …).
- **Baseline**: BKT, DKT (PyTorch); có chế độ smoke trong cấu hình chung.

---

## Cấu trúc thư mục (tóm lược)

| Thư mục / File | Vai trò |
|----------------|---------|
| `configs/global.yaml` | Siêu cấu hình: seed, chia dữ liệu, tham số đồ thị, sparse bins, baseline, audit. |
| `configs/*.yaml` | Theo từng bộ dữ liệu (vd. `kdd2010.yaml`, `assist2012.yaml`, `junyi.yaml`, `algebra2005.yaml`, `synthetic.yaml`). |
| `src/` | Pipeline Python (preprocess, split, Q-matrix, đồ thị, audit, baseline, báo cáo, figures). |
| `scripts/` | Lệnh ôn hoàn pipeline (`reproduce_one_dataset.sh`), master Python (`run_planA_master.py`), v.v. |
| `data/raw/`, `data/processed/` | Dữ liệu thô và đã xử lý (thường không đồng bộ lên repo đầy đủ — cần chuẩn bị theo README từng nguồn). |
| `results/` | Bảng CSV, báo cáo Markdown, artefact publication (`results/paper_ready/`). |
| `logs/` | Log chạy dài (`run_log_*`). |
| `P0_Master_Diagnostic_Report.md` | Ví dụ báo cáo tổng hợp (leakage, ablation, eco, …). |

---

## Yêu cầu môi trường

- **Python** 3.x (đường dẫn trình thông dịch trong `scripts/run_planA_master.py` hiện đang hard-code cho môi trường Windows cụ thể — nên sửa thành Python của máy bạn hoặc gọi `python -m …` như các script khác).
- Thư viện chính (tham khảo `scripts/run_planA_master.py` và mã `src/`):
  - `numpy`, `pandas`, `PyYAML`
  - `torch`, `scikit-learn`
  - `matplotlib`, `seaborn`, `tqdm`
  - Tùy bước: `openml`, `tabulate`

Cài ví dụ (tùy môi trường):

```bash
pip install numpy pandas pyyaml torch scikit-learn matplotlib seaborn tqdm openml tabulate
```

---

## Chạy nhanh (một bộ dữ liệu, một seed)

Từ thư mục gốc dự án, trên Bash (Git Bash/WSL/Linux):

```bash
bash scripts/reproduce_one_dataset.sh <tên_dataset> <seed_số>
# Ví dụ:
bash scripts/reproduce_one_dataset.sh kdd2010 2026
```

Trên PowerShell có thể gọi từng bước tương đương (cùng thứ tự như trong `scripts/reproduce_one_dataset.sh`): `python -m src.preprocess --config configs/<dataset>.yaml`, sau đó `split_checker`, `qmatrix_provenance`, `tri_relation_graph_builder`, các audit, v.v.

---

## Chạy automation nhiều seed / nhiều dataset

- `scripts/run_planA_master.py`: lặp theo dataset và seed (cần chỉnh `PYTHON_PATH`, `DATASETS`, `SEEDS` cho phù hợp máy của bạn).
- Shell hỗ trợ khác trong `scripts/`: `run_mini_test.sh`, `run_synthetic_minimal.sh`.

---

## Xuất bảng publication

Script `scripts/generate_publication_tables.py` và thư mục `results/paper_ready/` phục vụ gộp kết quả cho bản thảo báo.

---

## Ghi chú

- Logic merge cấu hình: `src.common.load_config` gộp `configs/global.yaml` với file cấu hình dataset.
- Các báo chi tiết theo dataset/fold nằm trong `results/reports/<dataset>/fold_0/` (DAG, leakage, eco, diagnostic, …).

---

## Tác giả / trích dẫn

Thêm thông tin tác giả, phiên bản và trích dẫn báo khi có bản thảo cuối cùng.
