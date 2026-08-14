# Passport OCR V4 — Verify-first Ground Truth Annotation

Mục tiêu của V4 là giảm tối đa thao tác annotate thủ công mà vẫn giữ ground truth độc lập với prediction.

## Ý tưởng chính

Thay vì nhập 8 field cho từng ảnh:

1. Pipeline final result được dùng để **pre-fill** 8 field.
2. Tool tự tạo **identity suggestion** bảo thủ.
3. Tool chọn khoảng `N` passport identities, ưu tiên hỗn hợp hard / medium / easy và đa dạng issuing country.
4. Mỗi identity chỉ cần annotate một **anchor image** (ảnh dễ đọc nhất trong group).
5. Nếu group identity có confidence cao, GUI mặc định cho phép **propagate GT** từ anchor sang toàn bộ image variants của passport đó.
6. Prediction không bao giờ được tự động coi là ground truth nếu chưa có human verification.

Kết quả: nếu 40 identities có trung bình 5–6 variants/identity, bạn có thể tạo GT cho khoảng 200–240 ảnh nhưng chỉ phải verify khoảng 40 anchor images.

---

## Laptop CPU có chạy được không?

Có. V4 annotation không chạy YOLO hoặc PaddleOCR.

Sau khi full pipeline đã chạy trên GPU, copy về laptop:

- repo code
- `outputs/`
- `input_images/` (khuyên copy để GUI xem ảnh gốc; nếu không có, GUI sẽ thử dùng transformed/VIZ/MRZ images trong outputs)

Không cần GPU cho prepare / GUI / export / benchmark reports.

GUI cần `tkinter` và `Pillow`. Windows Python chính thức thường có tkinter. Nếu thiếu Pillow:

```bash
pip install pillow
```

---

## 1. Tạo annotation queue

Mặc định chọn 40 identities:

```bash
python src/annotation_assistant.py prepare --target-identities 40
```

Output:

```text
ground_truth/
├── annotation_queue.csv
├── annotation_identity_selection.csv
└── identity_review_suggestions.csv   # chỉ tạo nếu có pair đáng review
```

`annotation_queue.csv` đã có prediction pre-filled vào các cột `gt_*`.

### Identity grouping

Auto propagation mặc định chỉ bật cho group `high` — thường là exact passport number có source/quality mạnh.

Group `medium` được gợi ý từ surname + DOB + expiry đủ chắc, nhưng GUI **không bật propagate mặc định**. Bạn có thể xem các variants rồi tự tick nếu xác nhận cùng passport.

Case không đủ chắc được giữ singleton để tránh merge sai identities.

---

## 2. Mở GUI annotate

```bash
python src/annotation_assistant.py gui
```

GUI hiển thị:

- ảnh gốc (nếu có), fallback sang processed output
- prediction của 8 field
- GT đã pre-fill bằng prediction
- quality / coverage / review reasons / source conflicts
- identity suggestion + confidence + group size
- nút lướt qua các variants trong cùng suggested identity

### Cách annotate nhanh nhất

Nếu toàn bộ prediction đúng:

**chỉ cần `Ctrl+Enter`** → verify + next.

Nếu có field sai:

- sửa đúng field đó trong cột Ground Truth
- `Ctrl+Enter`

Không phải gõ lại những field đã đúng.

### Keyboard shortcuts

- `Ctrl+Enter`: VERIFY + NEXT
- `Ctrl+S`: save hiện tại
- `Ctrl+R`: mark Needs Review
- `Alt+Left / Alt+Right`: identity trước / sau
- `Alt+Up / Alt+Down`: xem variant trước / sau trong cùng identity group

---

## 3. Xem tiến độ

```bash
python src/annotation_assistant.py status
```

Nếu benchmark đầu tiên chưa đủ và bạn muốn tăng từ 40 lên 60 identities **mà không mất annotation cũ**:

```bash
python src/annotation_assistant.py extend --target-identities 60
```

Tool sẽ giữ nguyên mọi GT đã verify/propagate và chỉ bổ sung identities mới.


Tool báo:

- số identities đã verify
- số identities còn lại
- số image samples đã có GT
- `Samples per manual anchor` — hệ số tiết kiệm thao tác nhờ propagation

---

## 4. Export Ground Truth

Khuyên dùng cho benchmark hiện tại: chia annotated identities thành 50% validation và 50% test, không cần train split.

```bash
python src/annotation_assistant.py export --assign-splits
```

Mặc định:

```text
train = 0%
val   = 50%
test  = 50%
```

Output:

```text
ground_truth/passport_ground_truth.csv
```

Nếu muốn split khác:

```bash
python src/annotation_assistant.py export --assign-splits --train 0.0 --val 0.6 --test 0.4
```

---

## 5. Chạy benchmark

```bash
python src/benchmark_suite.py
```

Sau đó tập trung vào:

```text
outputs/evaluation/split_metrics.csv
outputs/evaluation/failure_audit/failure_audit_summary.csv
```

Dùng `val` để quyết định thay đổi/tuning. Chỉ nhìn `test` khi muốn đánh giá phiên bản pipeline đã chốt.

---

## Ground-truth safety rules

V4 cố ý không có chức năng “auto accept high-confidence predictions” vì điều đó biến prediction thành label và làm benchmark thiên lệch.

Tự động hóa chỉ được dùng cho:

- pre-fill prediction để giảm gõ tay
- chọn sample/identity thông minh
- gợi ý identity group
- propagate GT **sau khi anchor đã được human verify**

Đây là điểm cân bằng giữa tốc độ annotate và độ tin cậy của benchmark.
