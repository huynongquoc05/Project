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

### 📑 Ví dụ kết quả phỏng vấn (JSON)

```json
{
  "_id": {
    "$oid": "68d46d430b751e74de781162"
  },
  "candidate_info": {
    "name": "Phạm Văn Nam,QTKD2",
    "profile": "Họ tên: Phạm Văn Nam, Lớp: QTKD2, Chuyên ngành: Marketing, Điểm chuyên cần: 6.4, Điểm 40%: 4.8, ",
    "classified_level": "yeu"
  },
  "interview_stats": {
    "timestamp": "2025-09-25T05:14:27.017412",
    "total_questions": 3,
    "final_score": 10,
    "topic": "Kiểu dữ liệu trong Java"
  },
  "question_history": [
    {
      "question_number": 1,
      "difficulty": "very_easy",
      "question": "Trong Java, có bao nhiêu kiểu dữ liệu cơ sở?",
      "answer": "Có 8 kiểu dữ liệu cơ sở trong java để lưu trữ các giá trị số nguyên, số thực, ký tự, đúng /sai. Thông tin thêm về kiểu dữ liệu cơ sở:\n-) Là kiểu dữ liệu đơn giản nhất trong Java. \n-) Tại một thời điểm, một kiểu dữ liệu cơ sở chỉ lưu trữ một giá trị đơn, không có các thông tin khác",
      "score": 10,
      "analysis": "Câu trả lời chính xác số lượng kiểu dữ liệu cơ sở (8) và phân loại đúng mục đích sử dụng của chúng (số nguyên, số thực, ký tự, đúng/sai). Ngoài ra, câu trả lời còn cung cấp thông tin bổ sung rất tốt về đặc điểm của kiểu dữ liệu cơ sở, thể hiện sự hiểu biết sâu sắc hơn về chủ đề này, phù hợp hoàn toàn với tài liệu tham khảo."
    },
    {
      "question_number": 2,
      "difficulty": "easy",
      "question": "Trong Java, bạn muốn khai báo một biến để lưu trữ giá trị nhiệt độ là `25.5`. Kiểu dữ liệu nào sẽ được sử dụng mặc định cho giá trị `25.5` này? Nếu bạn muốn lưu trữ giá trị này dưới dạng kiểu `float`, bạn sẽ khai báo nó như thế nào trong mã Java?",
      "answer": "Khiểu dữ liệu mặc định của Java cho giá trị này là double, nếu muốn khai báo dưới dạng kiểu float, cần thêm hậu tố \"f\" hoặc \"F\":\nfloat nhietDo=25.5f",
      "score": 10,
      "analysis": "Câu trả lời hoàn toàn chính xác và đầy đủ. Thí sinh đã nêu đúng kiểu dữ liệu mặc định cho số có dấu phẩy động là `double` và cách khai báo một giá trị `float` bằng cách thêm hậu tố `f` hoặc `F`, kèm theo một ví dụ mã rõ ràng. Điều này khớp hoàn toàn với tài liệu tham khảo."
    },
    {
      "question_number": 3,
      "difficulty": "medium",
      "question": "Trong Java, bạn có thể lưu trữ một số nguyên bằng cả kiểu dữ liệu cơ sở (`int`) và lớp gói (`Integer`). Hãy giải thích sự khác biệt cơ bản giữa `int` và `Integer`, và trong những tình huống ứng dụng thực tế nào bạn sẽ ưu tiên sử dụng `Integer` thay vì `int`?",
      "answer": "Trong Java, int là kiểu dữ liệu nguyên thủy (primitive type) với kích thước 32 bit, lưu trực tiếp giá trị số nguyên trong bộ nhớ, rất hiệu quả về tốc độ và bộ nhớ. Trong khi đó, Integer là lớp bao (Wrapper class) gói int thành một đối tượng, nhờ đó có thêm nhiều phương thức tiện ích như parseInt(), toHexString(), hoặc khả năng làm việc với các cấu trúc dữ liệu yêu cầu đối tượng (ví dụ ArrayList<Integer>). Điểm khác biệt quan trọng là int không thể lưu giá trị null, trong khi Integer có thể, nên phù hợp trong trường hợp cần biểu diễn dữ liệu có thể thiếu hoặc chưa xác định. Trong thực tế, bạn sẽ dùng int cho các phép tính số học cơ bản để đạt hiệu năng cao, còn dùng Integer khi cần tận dụng các phương thức hỗ trợ, cần làm việc với generic collection, hoặc cần giá trị null để thể hiện trạng thái đặc biệt",
      "score": 10,
      "analysis": "Câu trả lời hoàn toàn chính xác và đầy đủ. Thí sinh đã giải thích rõ ràng sự khác biệt cơ bản giữa `int` (kiểu nguyên thủy, hiệu quả về tốc độ và bộ nhớ, không thể null) và `Integer` (lớp gói, đối tượng, có phương thức tiện ích, có thể null, cần cho generic collection). Các tình huống ứng dụng thực tế được nêu ra cũng rất phù hợp và chính xác, thể hiện sự hiểu biết sâu sắc về cách sử dụng từng kiểu dữ liệu trong Java. Câu trả lời khớp hoàn toàn với tài liệu tham khảo và mở rộng thêm các kiến thức quan trọng."
    }
  ]
}
'''


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


