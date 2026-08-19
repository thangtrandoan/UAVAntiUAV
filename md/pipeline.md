# Kiến trúc End-to-End UAV Re-Identification Pipeline (Theo dòng thời gian)

Tài liệu này mô tả chi tiết luồng xử lý định danh sinh trắc học (ReID) của hệ thống UAVAntiUAV trong môi trường thực chiến, được trình bày theo trình tự thời gian của một video giám sát. Các module phát hiện (Detection) và bám sát (Tracking) cơ bản được tinh lược để tập trung hoàn toàn vào cốt lõi định danh (ReID).

## A. Công dụng và Lí do sử dụng các Module Cốt lõi

Trước khi đi vào dòng thời gian phân tích video, dưới đây là các module chính và lý do tại sao chúng được lựa chọn đưa vào pipeline:

0. **Tiền xử lý Ảnh (Image Preprocessing):**
   - **Công dụng:** Biến đổi khung hình video thô thành đầu vào chuẩn mực trước khi đưa vào mạng nơ-ron (Bao gồm: Cắt vùng Bounding Box, Resize về kích thước chuẩn 256x256, Chuẩn hóa Normalize giá trị pixel).
   - **Lí do sử dụng:** Mạng CNN yêu cầu đầu vào kích thước cố định. Việc cắt đúng vùng Bounding Box (do Object Detection cung cấp) giúp mạng loại bỏ nhiễu từ bối cảnh (bầu trời, cây cối), chỉ tập trung vào UAV. Chuẩn hóa pixel giúp tăng tốc độ tính toán và độ ổn định của mạng.

1. **Visual Backbone (CNN - GASNet/ResNet50-IBN):** 
   - **Công dụng:** Trích xuất đặc trưng không gian tĩnh (hình dáng, màu sắc, chi tiết khung sườn) của UAV từ từng frame ảnh đơn lẻ.
   - **Lí do sử dụng:** Mạng CNN vượt trội trong việc nắm bắt các chi tiết cục bộ (fine-grained) của vật thể. Biến thể có IBN (Instance-Batch Normalization) giúp hệ thống giữ được sự ổn định về nhận diện ngoại hình bất chấp sự thay đổi ánh sáng mạnh ngoài trời (chói nắng, bóng râm).

2. **Temporal Mamba Encoder:**
   - **Công dụng:** Rút trích đặc trưng thời gian (tốc độ, quỹ đạo bay, nhịp độ rung lắc cánh quạt) từ một chuỗi 16 frames liên tiếp.
   - **Lí do sử dụng:** Khác với Transformer tốn tài nguyên tính toán lũy thừa $O(N^2)$, kiến trúc State Space Model (Mamba) xử lý chuỗi với độ phức tạp tuyến tính $O(N)$. Điều này cho phép hệ thống "hiểu" thói quen chuyển động của UAV với tốc độ chớp nhoáng (<2ms), đáp ứng yêu cầu Real-time (thời gian thực) khắt khe trên các thiết bị Edge AI (như Jetson) thay vì phải dùng Server lớn.

3. **Ngân Hàng Ký Ức Đa Tầng (2-Tier Memory Bank):**
   - **Công dụng:** Quản lý và lưu trữ các vector đặc trưng của UAV làm mốc đối chiếu, chia làm 2 kho: Anchor (gốc bất biến) và Recent (30 frames gần nhất).
   - **Lí do sử dụng:** Khi bay, UAV liên tục thay đổi góc nhìn (xoay ngang, chúi mũi). Nếu chỉ dùng 1 vector trung bình sẽ làm "loãng" đặc trưng (Score Dilution). Cơ chế 2 tầng giúp hệ thống nhận ra UAV ở nhiều góc độ khác nhau, miễn là góc đó khớp với hình ảnh gốc hoặc những hình ảnh vừa mới ghi nhận được.

4. **Bộ Phạt Không Gian (Spatio-Temporal Penalty):**
   - **Công dụng:** Trừ điểm tương đồng ReID dựa trên khoảng cách vật lý của UAV giữa thời điểm mất dấu và thời điểm xuất hiện lại.
   - **Lí do sử dụng:** Đây là chốt chặn bảo mật cực kỳ quan trọng để chống lại chiến thuật "UAV mồi nhử" (Decoy). Kẻ địch có thể thả một UAV có ngoại hình y hệt quân ta ở cách đó rất xa. Bằng toán học cơ bản siêu nhẹ (O(1)), module này sẽ đánh rớt điểm các mục tiêu xuất hiện ở vị trí phi lý về mặt không gian/thời gian, ngăn chặn hệ thống bị lừa bám nhầm mục tiêu.

5. **Cơ chế Lọc Thô đến Tinh (Coarse-to-Fine ReID):**
   - **Công dụng:** Phân cấp quá trình xác thực UAV nghi ngờ thành 2 bước: quét nhanh 1 frame (Lọc Thô) và theo dõi sâu 16 frames (Lọc Tinh).
   - **Lí do sử dụng:** Khi mất track, có thể có hàng chục chiếc UAV nhiễu khác lọt vào khung hình. Việc chạy toàn bộ hệ thống phân tích sâu cho tất cả UAV là bất khả thi về mặt tài nguyên. Cơ chế Lọc Thô giúp loại bỏ ngay 90% UAV "rác" (khác màu, khác dáng) chỉ trong vài mili-giây, dồn toàn bộ sức mạnh tính toán để "chăm sóc" và đối chiếu kĩ lưỡng đúng chiếc UAV tình nghi cao nhất.

---

## 1. Giai đoạn T0: Khởi tạo và Lưu trữ Ký ức (Trước khi mất dấu)

Khi UAV mục tiêu (Target) lần đầu xuất hiện, được xác nhận và bám sát thành công, hệ thống bắt đầu quá trình trích xuất đặc trưng và xây dựng bộ nhớ định danh (Memory Bank).

*   **Tiền xử lý Dữ liệu (Data Preprocessing):** Trước khi đưa vào mạng, hình ảnh UAV được cắt (crop) ra khỏi khung hình video gốc một cách chính xác dựa trên tọa độ Bounding Box của thuật toán Detection. Tiếp theo, ảnh crop này được định dạng lại kích thước (Resize về 256x256 pixel) và chuẩn hóa giá trị pixel (Normalize) để triệt tiêu nhiễu màu sắc do cảm biến camera gây ra.
*   **Trích xuất Đặc trưng Không gian (Visual Backbone):** Mỗi khung hình (frame) đã qua tiền xử lý của UAV được đưa qua mạng nơ-ron CNN (như GASNet / ResNet50-IBN) để tạo ra một Vector Không gian 2560 chiều (bao gồm 2048 chiều Toàn cục đại diện cho tổng thể và 512 chiều Cục bộ đại diện cho chi tiết). 
*   **Trích xuất Đặc trưng Thời gian (Temporal Mamba Encoder):** Hệ thống tích lũy liên tục. Khi thu thập đủ một chuỗi 16 frames liên tiếp, chuỗi này được đưa qua Temporal Mamba Encoder để rút trích ra một Vector Thời gian 512 chiều, đại diện cho "thói quen chuyển động" và "độ rung lắc" của UAV.
*   **Dung hợp Đặc trưng (Feature Fusion):** Vector 2560 chiều và Vector 512 chiều được gộp lại, đi qua màng lọc khuếch đại thành một Vector Hoàn chỉnh 3072 chiều.
*   **Xây dựng Ngân Hàng Ký Ức (2-Tier Memory Bank):** 
    *   **Anchor Bank:** Lưu trữ Vector Hoàn chỉnh 3072 chiều đầu tiên làm "Hình dáng gốc ban đầu". Ký ức này là bất biến trong suốt vòng đời của đối tượng.
    *   **Recent Bank:** Liên tục cập nhật và lưu trữ các vector của 30 frames gần nhất. Điều này giúp hệ thống cập nhật các thay đổi ngoại hình khi UAV xoay góc hoặc đi qua vùng ánh sáng khác.

## 2. Giai đoạn T1: Trạng thái Mất dấu (Track Lost)

Khi UAV mục tiêu biến mất khỏi khung hình (bay ra sau tòa nhà, chui vào đám mây, hoặc bị mờ nhòe khiến Object Detection thất bại), bộ bám sát Tracking sẽ bị đứt gãy.

*   Hệ thống ghi nhận mục tiêu đã mất và chuyển trạng thái sang `LOST`.
*   Lưu lại tọa độ không gian cuối cùng (x, y) trước khi mục tiêu biến mất.
*   Toàn bộ bộ nhớ trong `Anchor Bank` và `Recent Bank` được khóa lại và bảo toàn nguyên vẹn.
*   Hệ thống chuyển sang "Chế độ Săn mồi", theo dõi các UAV mới xuất hiện (có thể là mục tiêu cũ, hoặc các UAV nhiễu khác) để chuẩn bị đối chiếu sinh trắc học.

## 3. Giai đoạn T2: Xuất hiện lại và Quá trình Tái định danh (Re-appearance & ReID)

Một hoặc nhiều UAV đột ngột xuất hiện lại trong khung hình (ví dụ: phát hiện 3 chiếc UAV xuất hiện cùng lúc). Hệ thống phải tìm ra đúng chiếc UAV mục tiêu cũ. Quá trình chọn lọc diễn ra qua 3 bước khắt khe (Coarse-to-Fine):

### Bước 3.1: Lọc Thô Siêu Tốc (Coarse ReID - Mili-giây đầu tiên)
*   Ngay tại frame đầu tiên các UAV xuất hiện, hệ thống chưa có đủ thời gian để lấy chuỗi chuyển động. Nó chỉ có 1 frame duy nhất cho mỗi "nghi phạm".
*   **Ảo ảnh chuyển động:** Hệ thống tự động nhân bản (duplicate) frame tĩnh này thành 16 frames ảo giống hệt nhau và ép chạy qua hệ thống Mamba.
*   So sánh kết quả với Memory Bank để tính ra **Điểm tương đồng Thô (Coarse Score)**.
*   **Loại trừ vĩnh viễn:** Bất kỳ UAV nào có điểm Coarse `< 0.50` sẽ bị đánh giá là khác biệt hoàn toàn (sai màu, sai kích thước) và bị loại bỏ ngay lập tức khỏi quy trình kiểm tra.
*   **Khóa tạm thời (Soft Lock):** Hệ thống chọn ra chiếc UAV có điểm Coarse cao nhất (phải `> 0.50`). Gán nhãn **SOFT LOCK**. Nòng súng/Camera tự động bám theo chiếc này để theo dõi chuyển động, chuẩn bị cho bước thẩm định kỹ hơn.

### Bước 3.2: Bộ Phạt Không Gian (Spatio-Temporal Penalty)
*   **Chống Mồi Nhử:** Địch có thể thả một UAV mồi nhử giống hệt quân ta ở khoảng cách rất xa. Hệ thống lập tức tính khoảng cách Euclidean từ tọa độ xuất hiện của "nghi phạm" đến tọa độ mất dấu cuối cùng ở Giai đoạn T1.
*   **Trừ điểm:** Khoảng cách càng bất hợp lý (quá xa trong thời gian ngắn), điểm ReID tổng hợp sẽ bị trừ càng nặng (với hệ số phạt `spatial_weight = 0.30`).
*   Một UAV dù có điểm ảnh giống 100% nhưng xuất hiện ở vị trí cách 2km chỉ sau 1 giây sẽ bị rớt điểm thê thảm.

### Bước 3.3: Xác Thực Chuyên Sâu (Fine ReID - Khoảng 0.5 giây sau)
*   Chiếc UAV đang bị Soft Lock sẽ được bám sát trong đúng 16 frames tiếp theo (tương đương khoảng 0.5s ở tốc độ 30 FPS).
*   Lúc này, hệ thống đã thu thập đủ chuyển động thực sự của cánh quạt và đường bay. Nó trích xuất lại Vector 3072 chiều hoàn chỉnh và tiến hành so khớp với Memory Bank để ra được **Điểm Tương đồng Tinh (Fine Score)**.
*   **Phán quyết cuối cùng:**
    *   **Thành công (Fine Score > 0.75):** Khẳng định chính xác 100% mục tiêu. Nâng cấp trạng thái thành **HARD LOCK**. Quá trình nối track hoàn tất, hệ thống trở lại bình thường.
    *   **Thất bại (Fine Score < 0.75):** Nhận diện nhầm (có thể do Coarse ReID bị lừa bởi hình dáng tĩnh). Hệ thống tống mục tiêu này vào Blacklist. Lập tức chuyển Soft Lock sang "nghi phạm" có điểm Coarse cao thứ 2 (nếu có) và lặp lại việc quan sát 16 frames.

## 4. Giai đoạn T3: Bám sát liên tục và Phòng thủ Anti-Hijack (Continuous Tracking)

Sau khi tái định danh thành công (Hard Lock), hệ thống tiếp tục bám sát, nhưng không hề chủ quan.

*   **Xác Thực Định Kỳ ngầm (Continuous Verification):** 
    *   Do các thuật toán Tracking cơ bản (như Kalman Filter) rất dễ bị "cướp mục tiêu" (Hijack) - ví dụ khi một con chim vô tình bay ngang cắt mặt UAV, bounding box có thể bị nhảy nhầm sang con chim.
    *   Để chống lại điều này, cứ mỗi chu kỳ 60 frames, hệ thống ReID sẽ lén lấy 16 frames hiện tại của mục tiêu để so khớp lại với `Anchor Bank` gốc ban đầu.
    *   Nếu điểm số bất ngờ rớt xuống `< 0.40` (chứng tỏ thứ đang bám theo không còn là mục tiêu gốc), hệ thống lập tức **Break Lock** (Bẻ gãy bám sát).
    *   Hệ thống chủ động quay ngược về trạng thái `LOST` (Giai đoạn T1) để ép tiến trình ReID lùng sục và bắt lại đúng chiếc UAV của mình.

---

## Bảng Ước Tính Chi Phí Tính Toán Sinh Trắc Học (ReID Computational Cost)

*Lưu ý: Các số liệu dưới đây là thời gian thực thi (Latency) ước tính trên các thiết bị Edge AI tiêu chuẩn như NVIDIA Jetson AGX Orin hoặc GPU rời RTX 3060. Không bao gồm chi phí tính toán của Object Detection.*

| Quá trình | Diễn giải Tác vụ | Chi phí ước tính (ms) | Đặc điểm tài nguyên |
| :--- | :--- | :--- | :--- |
| **1. Trích xuất Không gian (CNN)** | Chạy 1 frame qua Visual Backbone để lấy Vector 2560 chiều | **~5.0 ms / frame** | Chiếm tài nguyên chính, chạy trên từng bounding box |
| **2. Trích xuất Thời gian (Mamba)** | Chạy chuỗi 16 frames qua Temporal Encoder lấy Vector 512 chiều | **~1.0 - 2.0 ms / chuỗi** | Tốc độ cực cao, O(N) tuyến tính (đặc sản của Mamba) |
| **3. Feature Fusion & So khớp** | Ghép nối Vector và tính khoảng cách Cosine Distance | **< 0.1 ms / lượt** | Rất nhẹ, thuần phép tính ma trận mảng cơ bản |
| **4. Lọc Thô Siêu Tốc (Coarse ReID)** | Ép 1 frame của 1 UAV nhiễu qua quy trình (CNN + Mamba ảo) | **~6.0 ms / UAV** | Xử lý 10 UAV nhiễu cùng lúc chỉ tốn ~60ms |
| **5. Bộ Phạt Không Gian** | Tính Euclidean Distance giữa 2 tọa độ (x, y) | **~0.0001 ms / lượt** | O(1), tốc độ ngang bằng CPU xung nhịp cơ bản |
| **6. Xác Thực Chuyên Sâu (Fine ReID)** | Đợi 16 frames và tính toán ra điểm số chính xác cuối cùng | Khấu hao theo frame | Không gây giật lag vì tải trọng chia đều cho 16 khung hình |
| **7. Xác Thực Định Kỳ (Anti-Hijack)** | Chạy Mamba nền ngầm để kiểm tra lỗi mất mục tiêu | **~0.016 ms / frame** | Gần như vô hình (tốn ~1ms nhưng chia đều cho chu kỳ 60 frames) |
| **-> TỔNG TRỄ TÁI ĐỊNH DANH (ReID Latency)** | Tổng thời gian để hệ thống ra quyết định nhận diện UAV | **Dưới 10.0 ms** | Đảm bảo hiệu năng thời gian thực (Real-time FPS) ổn định |
