# 🤖 Adaptive Interviewer

Hệ thống AI Interviewer tự động, sử dụng **LangChain + FAISS + Gemini LLM** để tạo buổi phỏng vấn thích ứng:  
- Sinh câu hỏi theo **topic** và **độ khó phù hợp**.  
- Đánh giá câu trả lời (0–10 điểm).  
- Điều chỉnh độ khó theo năng lực thí sinh.  
- Xuất báo cáo tổng kết.  

---

## 📂 Danh sách file

| File | Mô tả |
|------|-------|
| **CreateVecto-intfloat-multilingual-e5-large-instruct.py** | Tạo vector database với model `intfloat/multilingual-e5-large-instruct`. |
| **NLTK.py** | Tạo vector database với model `intfloat/multilingual-e5-large-instruct` có sử dụng NLTK để tách chunk. |
| **CsVdataTest.py** | Tạo vector embedding từ file CSV điểm số thí sinh. |
| **intfloatmultilingual-e5-large-instruct.py** | Test truy vấn với model `intfloat/multilingual-e5-large-instruct`. |
| **LLM.py** | Sử dụng LLM Gemini dựa trên `vector_db2`. |
| **OthersModel.py** | Thử nghiệm tạo vector DBs với các model khác như `hiieu/halong_embedding`, `AITeamVN/Vietnamese_Embedding`, ... |
| **LLMInterviewer2_fixed.py** | Demo chương trình **AI Interviewer**. |

---

## 🚀 Thứ tự chạy file
1. Chạy `CreateVecto-intfloat-multilingual-e5-large-instruct.py` và `NLTK.py` để tạo vector database.  
2. Có thể chạy `intfloatmultilingual-e5-large-instruct.py` và `LLM.py` để truy vấn thử.  
3. Chạy `CsVdataTest.py` để tạo vector database điểm số.  
4. Cuối cùng chạy `LLMInterviewer2_fixed.py` để thực hiện phỏng vấn tự động.  

---

## 🎤 LLMInterviewer2_fixed.py

Chương trình **AI interviewer** tự động, dùng kiến thức từ vector DB + hồ sơ ứng viên để điều chỉnh độ khó câu hỏi theo thời gian thực, chấm điểm và đưa ra báo cáo cuối cùng.  

### 🧩 Các thành phần chính trong State Machine
- **Level (trình độ thí sinh)** – xác định từ điểm 40% trong hồ sơ:  
  - `yeu`, `trung_binh`, `kha`, `gioi`, `xuat_sac`.
- **QuestionDifficulty (độ khó câu hỏi)** – trạng thái động thay đổi trong quá trình phỏng vấn:  
  - `very_easy`, `easy`, `medium`, `hard`, `very_hard`.
- **Config (ngưỡng & luật)**  
  - Nếu điểm **>= 7** → lên `harder`.  
  - Nếu điểm **>= 4 và < 7** → giữ `same`.  
  - Nếu điểm **< 4** → xuống `easier`.  
  - Giới hạn:  
    - `MAX_ATTEMPTS_PER_LEVEL = 2`  
    - `MAX_TOTAL_QUESTIONS = 8`  
    - `MAX_UPPER_LEVEL = 2`  

---

### 🏗️ Ý nghĩa trạng thái & cách vận hành

**Level của thí sinh (`InterviewState.level`)**

- Quy định điểm xuất phát.  
- Ví dụ:  
  - `yeu` → bắt đầu từ `very_easy`.  
  - `kha` → bắt đầu từ `medium`.  
  - `gioi` → bắt đầu từ `hard`.  

**Level câu hỏi (`InterviewState.current_difficulty`)**

- Thay đổi động sau mỗi câu hỏi:  
  - Điểm **>=7** → nâng độ khó.  
  - Điểm **4–6.5** → giữ nguyên.  
  - Điểm **<4** → giảm độ khó.  

**Quy trình phỏng vấn**

1. Xác định level thí sinh → chọn độ khó khởi tạo.  
2. Sinh câu hỏi từ **FAISS + LLM**.  
3. Nhận câu trả lời.  
4. Chấm điểm + phân tích.  
5. Quyết định hành động tiếp theo (`harder/same/easier`).  
6. Lặp lại cho đến khi đạt điều kiện kết thúc.  
7. Xuất báo cáo tổng kết.  

---

### 🔚 Điều kiện kết thúc phỏng vấn

Quá trình sẽ dừng lại khi một trong các điều kiện sau xảy ra:

1. **Số câu hỏi ở cùng một độ khó đạt giới hạn**  
   - Mỗi độ khó chỉ cho phép tối đa **2 câu liên tiếp**.  

2. **Số lượng câu hỏi tổng cộng vượt ngưỡng**  
   - Buổi phỏng vấn không kéo dài quá **8 câu hỏi**.  

3. **Số lần nâng cấp độ khó vượt ngưỡng**  
   - Chỉ cho phép tăng độ khó tối đa **2 lần** so với ban đầu.  
   - Ví dụ: `easy → medium → hard`.  

4. **Người dùng chủ động dừng**  
   - Có thể nhấn `Ctrl + C` để kết thúc sớm.  

Khi kết thúc, hệ thống tổng hợp toàn bộ **lịch sử câu hỏi – trả lời – điểm số** và sinh **báo cáo tổng kết**.  


### 🔄 Flow tổng thể (State Machine)

```mermaid
stateDiagram-v2
    [*] --> LoadProfile: Bắt đầu
    LoadProfile --> ClassifyLevel: Lấy hồ sơ từ FAISS
    ClassifyLevel --> InitDifficulty: Xác định level thí sinh + độ khó ban đầu
    InitDifficulty --> AskQuestion
    
    state AskQuestion {
        [*] --> GenerateQ
        GenerateQ --> WaitAnswer: Sinh câu hỏi theo topic + độ khó
        WaitAnswer --> EvaluateAnswer: Nhận câu trả lời từ thí sinh
        EvaluateAnswer --> DecideAction: LLM chấm điểm + phân tích
        DecideAction --> UpdateState
    }
    
    UpdateState --> CheckEnd: Cập nhật số lần hỏi, độ khó mới
    CheckEnd --> AskQuestion: Nếu chưa kết thúc
    CheckEnd --> GenerateSummary: Nếu đã đủ điều kiện dừng
    
    GenerateSummary --> [*]


