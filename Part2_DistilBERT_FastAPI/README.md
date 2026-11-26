**# Customer Support Ticket Classifier**



**## Dataset Description**



**This project uses a synthetic customer support ticket dataset created specifically for this project.**



**The dataset contains 566 cleaned support tickets with three columns:**



**- text: Customer support ticket message**

**- category: Type of customer issue**

**- priority: Ticket priority level**



**## Dataset Source**



**Source: Original synthetic dataset generated for this project. No external dataset was used.**



**## Data Cleaning**



**- Converted ticket text to lowercase**

**- Removed leading and trailing spaces**

**- Removed duplicate tickets**

**- Checked category and priority class distribution**



**## Categories**



**- Login Issue**

**- Payment Issue**

**- Refund Request**

**- Account Issue**

**- Technical Issue**

**- Order Issue**



**## Priority Levels**



**- Low**

**- Medium**

**- High**



**## Dataset Split**



**The cleaned dataset was divided using stratified sampling:**



**- Training: 70%**

**- Validation: 15%**

**- Test: 15%**



**## Dataset Files**



**- `data/support\_tickets\_clean.csv`**

**- `data/train.csv`**

**- `data/validation.csv`**

**- `data/test.**



**Week 3 – Day 5**



**Task 1: Fine-tune DistilBERT and Compare**



**Model Comparison Table**



**Model| Accuracy| Macro F1| Training Time| Inference Time per Ticket**

**TF-IDF + Logistic Regression| 1.0000| 1.0000| 0.0324 s| 0.000021 s**

**DistilBERT| 1.0000| 1.0000| 290.45 s| 0.053588 s**



**Analysis**



**Both models achieved the same performance for category classification, with 1.0000 Accuracy and 1.0000 Macro F1 on the test set. However, the TF-IDF + Logistic Regression model was significantly faster to train and much faster during inference.**



**DistilBERT required substantially more computational time while providing no improvement in category classification performance on this dataset. Therefore, the baseline model is sufficient for this project.**



**DistilBERT was also tested for priority classification. It achieved 0.6000 Accuracy and 0.2500 Macro F1, while the TF-IDF baseline achieved 0.3647 Accuracy and 0.3198 Macro F1. The lower Macro F1 indicates that DistilBERT still struggled with the minority priority classes.**



**5-Line Model Selection Justification**



**1. Both TF-IDF + Logistic Regression and DistilBERT achieved 1.0000 Accuracy and 1.0000 Macro F1 on the category test set.**

**2. DistilBERT required significantly more training time than the TF-IDF baseline.**

**3. TF-IDF also provides much faster inference, making it more suitable for a real-time ticket classification API.**

**4. DistilBERT may become more useful with larger datasets or more complex language requiring stronger semantic understanding.**

**5. TF-IDF + Logistic Regression is selected for deployment because it provides the same performance with much lower cost and latency.**



**Final Model Selection**



**Selected model: TF-IDF + Logistic Regression**



**The TF-IDF baseline is selected for deployment because it provides excellent classification performance while being considerably faster and computationally cheaper than DistilBERT.**



## Task 2: FastAPI Endpoint and Demo

### API Features

- `POST /predict` accepts ticket text and returns category, priority, and confidence.
- `GET /health` checks whether the API is running.
- Models are loaded once at application startup.
- Pydantic validates the request input.
- Empty text returns HTTP 422.
- TestClient tests cover health check, valid prediction, empty text, and missing text.

### Run the API

From the project directory:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.main:app --reload
Submission branch: feature/week3-day5



