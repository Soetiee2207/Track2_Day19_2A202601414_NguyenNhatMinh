# Reflection — Lab 19

**Tên:** Nguyễn Nhật Minh
**Cohort:** _<A20-K3 >_
**Path đã chạy:** _<lite>_

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Trên golden set, mỗi kiểu query cho một kết quả khá dễ đoán nhưng cũng có điểm thú vị. Với query `exact`, BM25 thường tốt nhất vì các từ khóa trong câu hỏi xuất hiện gần như nguyên vẹn trong tài liệu. Với `paraphrase`, vector search có lợi thế hơn vì nó tìm theo ý nghĩa thay vì chỉ khớp từ. Còn với `mixed`, hybrid thường ổn định nhất: BM25 giữ được tín hiệu từ khóa, trong khi vector search bù lại những cách diễn đạt khác.

Tôi sẽ không dùng hybrid trong mọi trường hợp. Nếu người dùng tìm mã lỗi, tên sản phẩm, mã tài liệu hoặc một cụm từ chính xác, BM25 đơn thuần thường nhanh và dễ giải thích hơn. Ngược lại, với câu hỏi tự nhiên, nhiều cách diễn đạt hoặc nội dung đa ngôn ngữ, pure vector có thể phù hợp hơn. Hybrid đáng dùng khi chưa biết trước người dùng sẽ viết query theo kiểu nào, dù phải chấp nhận thêm chi phí index và latency.

---

## Điều ngạc nhiên nhất khi làm lab này

Điều bất ngờ nhất là embedding model ảnh hưởng rất rõ đến kết quả tiếng Việt. Một hệ thống chạy bình thường và trả về kết quả có vẻ hợp lý chưa chắc đã tìm đúng ý người dùng.

---

## Bonus challenge

- [X] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_
