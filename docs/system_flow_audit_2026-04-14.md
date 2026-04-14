# Báo cáo rà soát luồng hệ thống OMR (2026-04-14)

## Mục tiêu
Rà soát các luồng nghiệp vụ có nguy cơ gây sai lệch dữ liệu, mất nhất quán hoặc khó vận hành; đề xuất phương án tối ưu hóa theo hướng **sạch – nhất quán – đúng đắn dữ liệu**.

## Các luồng chưa hợp lý và hướng xử lý

### 1) Luồng import đáp án: thông điệp và xử lý không nhất quán cho True/False
- Hiện trạng:
  - Nhánh import bảng 1 cột `Answer` chỉ parse MCQ hoặc Numeric.
  - Thông báo lỗi lại nói rằng định dạng hợp lệ gồm cả TF 4 ký tự.
- Rủi ro:
  - Người dùng nhập TF hợp lệ nhưng bị báo lỗi “không hợp lệ”, tạo cảm giác hệ thống sai.
  - Dễ phát sinh thao tác workaround và dữ liệu chấm không đúng kỳ vọng.
- Tối ưu đề xuất:
  - Đồng bộ parser + thông báo: hoặc thực sự hỗ trợ TF ở nhánh `Answer`, hoặc bỏ TF khỏi thông báo.
  - Bổ sung test hồi quy riêng cho `Answer=TF`.

### 2) Luồng replace dữ liệu theo subject là “xóa trước, ghi sau”
- Hiện trạng:
  - `replace_answer_keys_for_subject` và `replace_scan_results_for_subject` đều `DELETE` trước rồi `INSERT` lại.
  - Không có transaction bao trùm để đảm bảo atomicity xuyên suốt thao tác replace.
- Rủi ro:
  - Nếu lỗi giữa chừng (crash, exception, mất điện), subject có thể bị rỗng dữ liệu.
  - Mất tính toàn vẹn khi thao tác khối lượng lớn.
- Tối ưu đề xuất:
  - Bọc bằng transaction rõ ràng (`BEGIN IMMEDIATE ... COMMIT/ROLLBACK`).
  - Dùng bảng tạm + swap, hoặc upsert theo batch có kiểm soát phiên bản.

### 3) Luồng upsert scan result chưa có ràng buộc unique ở tầng DB
- Hiện trạng:
  - `upsert_scan_result` tự `SELECT` theo `(subject_key, image_path)` rồi `UPDATE/INSERT` thủ công.
  - Schema không có `UNIQUE(subject_key, image_path)` cho `scan_results`.
- Rủi ro:
  - Dễ sinh bản ghi trùng cùng ảnh/môn khi có cạnh tranh ghi hoặc gọi lặp.
  - Truy vấn “lấy bản ghi mới nhất” gây khó tái lập và debug.
- Tối ưu đề xuất:
  - Thêm unique constraint + dùng `INSERT ... ON CONFLICT DO UPDATE`.
  - Định nghĩa khóa nghiệp vụ rõ ràng cho kết quả quét.

### 4) Luồng xóa phiên thi chưa dọn dữ liệu liên quan
- Hiện trạng:
  - `delete_exam_session` chỉ xóa bảng `exams`, không cascade xóa `scan_results`, `scores`, `recheck_history` liên quan.
- Rủi ro:
  - Dữ liệu mồ côi tích lũy, báo cáo/histories có thể lệch ngữ cảnh phiên thi.
- Tối ưu đề xuất:
  - Thiết kế quan hệ session-centric (thêm `session_id` vào bảng nghiệp vụ) + foreign key `ON DELETE CASCADE`.
  - Hoặc tạo service xóa theo domain (xóa phiên + các subject thuộc phiên + scan/score/recheck).

### 5) Luồng đồng bộ đáp án nhận dạng có cơ chế “căn theo vị trí” khi lệch số câu
- Hiện trạng:
  - `_aligned_marked_answers` map đáp án theo index khi không khớp trực tiếp số câu.
- Rủi ro:
  - Có thể “dịch cột” im lặng (Q2 nhận dữ liệu của Q3), dẫn đến chấm sai nhưng khó phát hiện.
- Tối ưu đề xuất:
  - Chỉ cho phép map theo khóa câu hỏi tuyệt đối; nếu lệch thì phát sinh lỗi cấu hình rõ ràng.
  - Tách chế độ “repair mapping” thành thao tác thủ công có xác nhận người dùng.

### 6) Luồng chấm theo phần (TF) có nguy cơ vượt tổng điểm phần
- Hiện trạng:
  - Điểm TF được cộng theo rule 1/2/3/4 đúng cho **mỗi câu TF**.
  - Nếu cấu hình rule không được chuẩn hóa theo tổng số câu, tổng TF thực tế có thể vượt `section_scores.TF.total_points`.
- Rủi ro:
  - Tổng điểm môn không ổn định giữa các mã đề/đợt nhập cấu hình.
- Tối ưu đề xuất:
  - Chuẩn hóa rule TF theo tổng điểm phần và số câu TF tại thời điểm lưu cấu hình.
  - Áp dụng hậu kiểm: nếu tổng tối đa lý thuyết > tổng phần thì cảnh báo/khóa lưu.

## Chuẩn hóa kiến trúc dữ liệu đề xuất
1. **Xác định khóa nghiệp vụ cứng** cho từng thực thể (`session_id`, `subject_key`, `exam_code`, `student_id`, `image_path`).
2. **Ràng buộc DB trước, code sau**: unique + foreign key + cascade + chỉ mục đúng chiều truy vấn.
3. **Transaction cho mọi thao tác batch/destructive**.
4. **Idempotent pipelines**: chạy lại không tạo bản ghi trùng.
5. **Bắt buộc validation ở biên nhập liệu** (import file, edit tay, API mapping).
6. **Audit log theo event domain** (before/after, actor, reason) để truy vết dễ.

## Kế hoạch triển khai gợi ý (ưu tiên)
- P1: Sửa import Answer/TF + test hồi quy.
- P1: Thêm unique constraint scan_results + refactor upsert ON CONFLICT.
- P1: Transaction hóa replace_*.
- P2: Thiết kế lại quan hệ session và cleanup cascade.
- P2: Ràng buộc scoring TF theo tổng điểm phần.
- P3: Bổ sung health-check dữ liệu định kỳ (trùng, mồ côi, thiếu key).
