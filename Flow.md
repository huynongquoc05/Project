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
