# Báo cáo Tổng hợp Cuối cùng Bài báo P0 (Master Report)

> Được tạo tự động dựa trên toàn bộ 144 vòng huấn luyện mô hình.

## 1. Kết quả Kiểm toán Rò rỉ (Leakage Audit Summary L1-L5)
| Dataset | L1_edge | L2_qmatrix | L3_temporal | L4_coldstart | L5_co_occurrence |
|---|---|---|---|---|---|
| kdd2010 | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| junyi | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| assist2012 | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |

## 2. Kết quả Cắt tỉa Quan hệ Đồ thị (Relation Ablation Results)
Bảng dưới đây là trung bình (mean) và độ lệch chuẩn (std) của AUC/ACC trên 3 seeds.

| Dataset    | Model    | Graph Variant    |   AUC (Mean) |   AUC (Std) |   ACC (Mean) |   ACC (Std) |
|:-----------|:---------|:-----------------|-------------:|------------:|-------------:|------------:|
| assist2012 | DKT      | no_graph         |       0.6075 |      0.0002 |       0.6875 |      0.0045 |
| assist2012 | DKT      | E_pre            |       0.6067 |      0.0027 |       0.686  |      0.0002 |
| assist2012 | DKT      | E_pre_E_sim      |       0.607  |      0.002  |       0.688  |      0.0016 |
| assist2012 | DKT      | E_pre_E_sim_E_co |       0.6065 |      0.0021 |       0.6884 |      0.0046 |
| assist2012 | GIKT     | no_graph         |       0.5517 |      0      |       0.7067 |      0      |
| assist2012 | GIKT     | E_pre            |       0.5517 |      0      |       0.7067 |      0      |
| assist2012 | GIKT     | E_pre_E_sim      |       0.5517 |      0      |       0.7067 |      0      |
| assist2012 | GIKT     | E_pre_E_sim_E_co |       0.5517 |      0      |       0.7066 |      0      |
| assist2012 | GKT      | no_graph         |       0.5517 |      0      |       0.7067 |      0      |
| assist2012 | GKT      | E_pre            |       0.5517 |      0      |       0.7067 |      0      |
| assist2012 | GKT      | E_pre_E_sim      |       0.5517 |      0      |       0.7067 |      0      |
| assist2012 | GKT      | E_pre_E_sim_E_co |       0.5517 |      0      |       0.7066 |      0      |
| assist2012 | simpleKT | no_graph         |       0.5517 |      0      |       0.7067 |      0      |
| assist2012 | simpleKT | E_pre            |       0.5517 |      0      |       0.7067 |      0      |
| assist2012 | simpleKT | E_pre_E_sim      |       0.5517 |      0      |       0.7067 |      0      |
| assist2012 | simpleKT | E_pre_E_sim_E_co |       0.5517 |      0      |       0.7066 |      0      |
| junyi      | DKT      | no_graph         |       0.6685 |      0.0007 |       0.7063 |      0.0003 |
| junyi      | DKT      | E_pre            |       0.669  |      0.0006 |       0.706  |      0.0007 |
| junyi      | DKT      | E_pre_E_sim      |       0.6687 |      0.0003 |       0.7065 |      0.0002 |
| junyi      | DKT      | E_pre_E_sim_E_co |       0.669  |      0.0008 |       0.7062 |      0.0006 |
| junyi      | GIKT     | no_graph         |       0.6075 |      0      |       0.6882 |      0      |
| junyi      | GIKT     | E_pre            |       0.6075 |      0      |       0.6882 |      0      |
| junyi      | GIKT     | E_pre_E_sim      |       0.6075 |      0      |       0.6882 |      0      |
| junyi      | GIKT     | E_pre_E_sim_E_co |       0.6076 |      0      |       0.6882 |      0      |
| junyi      | GKT      | no_graph         |       0.6075 |      0      |       0.6882 |      0      |
| junyi      | GKT      | E_pre            |       0.6075 |      0      |       0.6882 |      0      |
| junyi      | GKT      | E_pre_E_sim      |       0.6075 |      0      |       0.6882 |      0      |
| junyi      | GKT      | E_pre_E_sim_E_co |       0.6076 |      0      |       0.6882 |      0      |
| junyi      | simpleKT | no_graph         |       0.6075 |      0      |       0.6882 |      0      |
| junyi      | simpleKT | E_pre            |       0.6075 |      0      |       0.6882 |      0      |
| junyi      | simpleKT | E_pre_E_sim      |       0.6075 |      0      |       0.6882 |      0      |
| junyi      | simpleKT | E_pre_E_sim_E_co |       0.6076 |      0      |       0.6882 |      0      |
| kdd2010    | DKT      | no_graph         |       0.7276 |      0.0164 |       0.7698 |      0.0178 |
| kdd2010    | DKT      | E_pre            |       0.702  |      0.0134 |       0.7675 |      0.0245 |
| kdd2010    | DKT      | E_pre_E_sim      |       0.7241 |      0.0114 |       0.7779 |      0.0192 |
| kdd2010    | DKT      | E_pre_E_sim_E_co |       0.7194 |      0.0097 |       0.7762 |      0.0146 |
| kdd2010    | GIKT     | no_graph         |       0.7403 |      0      |       0.7129 |      0      |
| kdd2010    | GIKT     | E_pre            |       0.7403 |      0      |       0.7129 |      0      |
| kdd2010    | GIKT     | E_pre_E_sim      |       0.7453 |      0      |       0.7233 |      0      |
| kdd2010    | GIKT     | E_pre_E_sim_E_co |       0.7399 |      0      |       0.7121 |      0      |
| kdd2010    | GKT      | no_graph         |       0.7403 |      0      |       0.7129 |      0      |
| kdd2010    | GKT      | E_pre            |       0.7403 |      0      |       0.7129 |      0      |
| kdd2010    | GKT      | E_pre_E_sim      |       0.7453 |      0      |       0.7233 |      0      |
| kdd2010    | GKT      | E_pre_E_sim_E_co |       0.7399 |      0      |       0.7121 |      0      |
| kdd2010    | simpleKT | no_graph         |       0.7403 |      0      |       0.7129 |      0      |
| kdd2010    | simpleKT | E_pre            |       0.7403 |      0      |       0.7129 |      0      |
| kdd2010    | simpleKT | E_pre_E_sim      |       0.7453 |      0      |       0.7233 |      0      |
| kdd2010    | simpleKT | E_pre_E_sim_E_co |       0.7399 |      0      |       0.7121 |      0      |


---

## 3. Eco Provenance Audit Module

Bảng dưới đây thống kê các thuộc tính của đồ thị đồng xuất hiện (E_co) bao gồm độ bao phủ trên các kỹ năng thưa thớt (Sparse-KC).

| Dataset    | Symmetry Pass   | Train-only Support   | Weight Dist (Mean ± Std)   | Sparse-KC Coverage   |
|:-----------|:----------------|:---------------------|:---------------------------|:---------------------|
| kdd2010    | True            | True                 | 0.422 ± 0.385              | 77.78% (28/36)       |
| junyi      | True            | True                 | 0.794 ± 0.745              | 100.00% (3/3)        |
| assist2012 | True            | True                 | 1.213 ± 0.932              | 86.90% (73/84)       |

---

## Table 7: Full Ablation for simpleKT and GIKT (Supplementary Material)

> Lời tựa: Dưới đây là số liệu ablation chi tiết cho các mô hình simpleKT và GIKT được dùng làm sanitiy check.

| Dataset    | Model    | Graph Variant    |   AUC (Mean) |   AUC (Std) |   ACC (Mean) |   ACC (Std) |
|:-----------|:---------|:-----------------|-------------:|------------:|-------------:|------------:|
| assist2012 | GIKT     | no_graph         |       0.5517 |           0 |       0.7067 |           0 |
| assist2012 | GIKT     | E_pre            |       0.5517 |           0 |       0.7067 |           0 |
| assist2012 | GIKT     | E_pre_E_sim      |       0.5517 |           0 |       0.7067 |           0 |
| assist2012 | GIKT     | E_pre_E_sim_E_co |       0.5517 |           0 |       0.7066 |           0 |
| assist2012 | simpleKT | no_graph         |       0.5517 |           0 |       0.7067 |           0 |
| assist2012 | simpleKT | E_pre            |       0.5517 |           0 |       0.7067 |           0 |
| assist2012 | simpleKT | E_pre_E_sim      |       0.5517 |           0 |       0.7067 |           0 |
| assist2012 | simpleKT | E_pre_E_sim_E_co |       0.5517 |           0 |       0.7066 |           0 |
| junyi      | GIKT     | no_graph         |       0.6075 |           0 |       0.6882 |           0 |
| junyi      | GIKT     | E_pre            |       0.6075 |           0 |       0.6882 |           0 |
| junyi      | GIKT     | E_pre_E_sim      |       0.6075 |           0 |       0.6882 |           0 |
| junyi      | GIKT     | E_pre_E_sim_E_co |       0.6076 |           0 |       0.6882 |           0 |
| junyi      | simpleKT | no_graph         |       0.6075 |           0 |       0.6882 |           0 |
| junyi      | simpleKT | E_pre            |       0.6075 |           0 |       0.6882 |           0 |
| junyi      | simpleKT | E_pre_E_sim      |       0.6075 |           0 |       0.6882 |           0 |
| junyi      | simpleKT | E_pre_E_sim_E_co |       0.6076 |           0 |       0.6882 |           0 |
| kdd2010    | GIKT     | no_graph         |       0.7403 |           0 |       0.7129 |           0 |
| kdd2010    | GIKT     | E_pre            |       0.7403 |           0 |       0.7129 |           0 |
| kdd2010    | GIKT     | E_pre_E_sim      |       0.7453 |           0 |       0.7233 |           0 |
| kdd2010    | GIKT     | E_pre_E_sim_E_co |       0.7399 |           0 |       0.7121 |           0 |
| kdd2010    | simpleKT | no_graph         |       0.7403 |           0 |       0.7129 |           0 |
| kdd2010    | simpleKT | E_pre            |       0.7403 |           0 |       0.7129 |           0 |
| kdd2010    | simpleKT | E_pre_E_sim      |       0.7453 |           0 |       0.7233 |           0 |
| kdd2010    | simpleKT | E_pre_E_sim_E_co |       0.7399 |           0 |       0.7121 |           0 |

