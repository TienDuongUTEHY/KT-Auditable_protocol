# Rà soát dự án P0 đối chiếu thiết kế `P0_DUong.pdf`

Tài liệu đối chiếu: Roadmap P0 KTr-GK-CL v3.0 (phần thí nghiệm E0–E8, artefact, baseline, checklist figure/table trong PDF).

**Phạm vi:** Đánh giá khoảng cách giữa repo và thiết kế; không bao gồm chỉnh sửa mã trong phiên làm việc tạo file này.

---

## 1. Thí nghiệm bắt buộc (E0–E5)

### E0 — Thống kê dataset

**Thiết kế:** Junyi + ASSIST2012 + KDD Cup 2010; `#learners`, `#questions`, `#KC`, `#interactions`, độ dài chuỗi trung bình, phân phối tần suất KC.

**Repo:** `preprocess` ghi `dataset_stats.csv` (Learners, Questions, Skills, Interactions, AvgSeqLen, QMatrixSource) — đáp ứng phần lớn.

**Chưa đủ / cần bổ sung:**

- Phân phối tần suất KC (bảng hoặc histogram chuẩn cho paper) chưa thấy được xuất có hệ thống như mục E0; có thể cần module xuất từ `interactions.csv` / train split.
- `scripts/generate_publication_tables.py` xây Table 1 từ `sparse_skill_summary` và ước lượng Interactions gần đúng — lệch với thống kê chuẩn từ `dataset_stats.csv` (thiết kế Table 1 trong PDF).

---

### E1 — Five-type leakage audit

**Thiết kế:** Không overlap learner; cạnh đồ thị train-only cho cả ba loại; đồ thị theo fold; co-occurrence train-only; bảng 5 loại.

**Repo:** Có `leakage_audit_log.csv` + `leakage_audit_report.md`; split theo learner (không overlap learner giữa train/valid/test). L1/L5 dựa trên cột `source_split`; L2 trên nguồn Q-matrix.

**Chưa đủ / cần sửa logic (khi triển khai sau):**

- L3 (temporal) và L4 (cold-start) đang luôn PASS (placeholder) — chưa đáp ứng tinh thần “audit có ý nghĩa” trong thiết kế.
- Chưa thấy kiểm tra tường minh kiểu: test edge không dùng thông tin ngoài train, cold-start neighborhood theo định nghĩa protocol (cần bám đúng định nghĩa L1–L5 trong tài liệu đầy đủ).
- No learner overlap được đảm bảo bởi cách chia, nhưng chưa ghi nhận thành một dòng check trong leakage log (thiết kế nhấn mạnh điều này).

---

### E2 — Tri-relation graph audit

**Thiết kế:** `#E_pre`, `#E_sim`, `#E_co`, degree, density, components, cycles before/after pruning, bảng audit.

**Repo:** `graph_stats.csv` + `dag_report.md` / `dag_audit.csv` (cycles before/after, prune).

**Chưa đủ:**

- `graph_statistics.py` không xuất số connected components (hoặc tương đương per relation) — thiết kế liệt kê rõ “components”.
- Một số dòng lỗi trong `graph_stats` khi thiếu file dùng `edge_type` kiểu tên file thô — dễ lẫn với dòng chuẩn (cần làm sạch schema bảng cho paper).

---

### E3 — E_co quality (KDD Cup 2010 + Junyi)

**Thiết kế:** Phân phối PMI/count, symmetry pass, sparse-skill coverage theo stratum.

**Repo:** `eco_audit.md` / `eco_audit.csv` (symmetry, train-only); `make_figures` có histogram/CDF trọng số E_co.

**Chưa đủ:**

- `eco_audit` chưa tích hợp sparse-skill coverage theo stratum như E3; cần một module / một bảng artefact thống nhất với định nghĩa stratum trong protocol.
- Phân phối count (song song PMI) chưa thấy được nhấn mạnh trong artefact `eco_audit` giống mô tả thiết kế.

---

### E4 — Sparse-skill diagnostic

**Thiết kế:** Frequency strata, graph coverage theo stratum, baseline AUC/ACC theo stratum.

**Repo:** `sparse_skill_profile.csv` / `.md` với strata theo phân vị p33/p66 (sparse / medium / dense), có figure 3.

**Chưa đủ:**

- `configs/global.yaml` định nghĩa bins (very_sparse, sparse, medium, frequent theo ngưỡng count) không được dùng trong `sparse_skill_profile.py` — lệch protocol giữa config và code.
- Không có baseline theo từng stratum (E4 bắt buộc trong PDF).
- Graph coverage per stratum chưa được tính/ghép trong profile (chỉ có thống kê kỹ năng/tương tác).

---

### E5 — Relation ablation (1–2 dataset; simpleKT hoặc DKT)

**Thiết kế:** `no_graph` vs `E_pre` vs `E_pre+E_sim` vs full; một strong baseline hiện đại (simpleKT hoặc AKT).

**Repo:** Có ablation 4 variant với DKT (thật) và BKT (proxy logistic). Có nhánh tên SIMPLEKT / GKT / GIKT nhưng cùng chạy `run_bkt_proxy` — không phải simpleKT/GKT/GIKT thật. Không có AKT.

**Chưa đủ (nghiêm trọng so với E5 + mục 7.2 PDF):**

- Thiếu strong baseline đúng nghĩa (pyKT simpleKT hoặc AKT như gợi ý thiết kế).
- Pipeline mặc định (`reproduce_one_dataset.sh`, `global.yaml`) chỉ BKT + DKT — chưa đáp ứng combo “classic + strong” mà PDF mô tả.
- Cấu hình `baseline.smoke_mode`, `max_epochs: 5` — chỉ phù hợp smoke, không đủ cho bảng ablation bắt buộc nếu venue yêu cầu huấn luyện đủ nghiêm.

---

## 2. Thí nghiệm tùy chọn (E6–E8)

| Mục | Thiết kế                         | Repo |
|-----|----------------------------------|------|
| **E6** | DDR (Junyi, cite Tuấn); line chart / heatmap | Chưa thấy module hoặc figure DDR. |
| **E7** | Độ nhạy ngưỡng E_co (KDD); heatmap `k_co`, PMI vs count | Chưa thấy sweep tham số / heatmap. |
| **E8** | Sanity GKT/GIKT với tri-graph   | Tên model có trong CLI nhưng không phải GKT/GIKT thật. |

---

## 3. Dataset theo PDF (mục 7.1)

- **Bắt buộc:** ASSIST2012, Junyi, KDD2010 — repo có config tương ứng.
- **Tùy chọn:** XES3G5M — chưa có pipeline.
- **EdNet KT1:** không bắt buộc — nhất quán với “không làm”.
- Repo thêm **algebra2005**, **synthetic** — không trái PDF (mở rộng nội bộ); cần tránh lẫn story nếu paper chỉ gắn 3 dataset chính.

---

## 4. Checklist artefact (mục 8 PDF) vs repo

Hầu hết đường dẫn / tên file đã có: `configs/*.yaml`, `split_hash.txt`, `qmatrix_provenance.md`, `edge_provenance.csv`, `leakage_audit_log.csv`, `graph_stats.csv`, `eco_audit.md`, `sparse_skill_profile.md`, `baseline_results.csv`, `reproduce_one_dataset.sh`, README (repo dùng `README.MD` — khác chữ hoa/thường so với `readme.md` trong PDF; thường không ảnh hưởng GitHub).

**Chất lượng nội dung artefact so với mô tả PDF:**

- **`qmatrix_provenance.md`:** rất tối giản (source + PASS/FAIL); thiết kế nhắc quy tắc train-only khi Q derived — cần mở rộng nếu Q có nhánh suy diễn từ train.
- **`common.py`:** hàm hash file/DataFrame kiểu mock — nếu provenance thật dựa vào đây sẽ không đáp ứng mục reproducibility / edge provenance có băm thật.
- **`run_planA_master.py`:** `PYTHON_PATH` hard-code máy cụ thể — trái tinh thần repo cho supervisor/reviewer chạy lại (PDF nhấn config + log + minimal reproduce).

---

## 5. Bảng hình mục tiêu (danh sách trong PDF)

- **Figure 1** pipeline: code có `fig1_pipeline.pdf` (tổng quan + edge counts) — gần “overview”; có thể chỉnh graphic cho đúng story “leakage-controlled construction”.
- **Table 1** dataset + graph availability: cần đồng bộ nguồn số liệu (script publication hiện dùng nguồn chưa tối ưu).
- **Table 2** leakage checklist: có log CSV nhưng cần bảng paper-ready và check thật (L3/L4).
- **Table 3** tri-relation audit: có `graph_stats` + DAG — thiếu components như đã nêu.
- **Figure 2** E_co quality: có phân phối weight — thiếu sparse coverage đồng bộ với E3.
- **Figure 3** sparse strata: có — thiếu baseline theo stratum.
- **Table 4** ablation: có dữ liệu DKT/BKT — thiếu simpleKT/AKT đúng nghĩa.
- **Figure 4–5** (DDR, sensitivity): chưa có trong code như mô tả optional.

---

## 6. Tóm tắt ưu tiên triển khai (đề xuất)

1. **E5 + §7.2:** Tích hợp simpleKT hoặc AKT (pyKT) và chạy ablation đúng protocol; tách rõ “diagnostic smoke” vs “số liệu paper”.
2. **E4 + config:** Thống nhất định nghĩa stratum (`sparse_skill` trong YAML vs code); thêm metric theo stratum + graph coverage theo stratum.
3. **E1:** Thay placeholder L3/L4 bằng kiểm tra theo định nghĩa protocol; ghi rõ learner disjoint trong log.
4. **E2:** Bổ sung số component (và có thể so sánh before/after prune nếu protocol yêu cầu).
5. **E3:** Mở rộng `eco_audit` với coverage theo stratum + phân phối count.
6. **Reproducibility:** Bỏ hard-code Python path; provenance hash thật; Table 1 lấy từ `dataset_stats` + phân phối KC.
7. **Optional E6–E7:** Module DDR (trích dẫn Tuấn) và grid `k_co` — nếu muốn khớp roadmap đầy đủ.

---

## Ghi chú

Đối chiếu dựa trên nội dung text trích từ `P0_DUong.pdf` trong workspace (một số dòng PDF bị lỗi font khi extract). Nếu có phiên bản PDF đầy đủ định nghĩa L1–L5, DDR, và cold-start chi tiết, nên đối chiếu lại từng điều kiện PASS/FAIL cho sát protocol.
