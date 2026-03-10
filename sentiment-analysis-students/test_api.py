<<<<<<< HEAD
# test_api.py - Script để test API
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("\n" + "="*80)
    print("1. TEST HEALTH CHECK")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_single_prediction():
    """Test single prediction"""
    print("\n" + "="*80)
    print("2. TEST SINGLE PREDICTION")
    print("="*80)
    
    test_cases = [
        "Giảng viên dạy rất hay và dễ hiểu",
        "Cơ sở vật chất trường còn thiếu thốn",
        "Chương trình đào tạo cần cập nhật thêm",
        "Thầy giảng bài không rõ ràng",
        "Phòng học sạch sẽ, thoáng mát"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n--- Test case {i} ---")
        print(f"Input: {text}")
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Sentiment: {result['sentiment']['sentiment']}", end="")
            if result['sentiment']['confidence']:
                print(f" (confidence: {result['sentiment']['confidence']:.2%})")
            else:
                print()
            print(f"Topic: {result['topic']['topic']}", end="")
            if result['topic']['confidence']:
                print(f" (confidence: {result['topic']['confidence']:.2%})")
            else:
                print()
        else:
            print(f"Error: {response.status_code} - {response.text}")

def test_sentiment_only():
    """Test sentiment-only prediction"""
    print("\n" + "="*80)
    print("3. TEST SENTIMENT ONLY")
    print("="*80)
    
    text = "Giảng viên nhiệt tình, tận tâm với sinh viên"
    print(f"Input: {text}")
    
    response = requests.post(
        f"{BASE_URL}/predict/sentiment",
        json={"text": text}
    )
    
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_topic_only():
    """Test topic-only prediction"""
    print("\n" + "="*80)
    print("4. TEST TOPIC ONLY")
    print("="*80)
    
    text = "Phòng thí nghiệm thiếu thiết bị"
    print(f"Input: {text}")
    
    response = requests.post(
        f"{BASE_URL}/predict/topic",
        json={"text": text}
    )
    
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "="*80)
    print("5. TEST BATCH PREDICTION")
    print("="*80)
    
    texts = [
        "Giảng viên dạy hay",
        "Trường cần đầu tư thêm máy tính",
        "Chương trình học quá khó",
        "Thư viện thiếu sách tham khảo"
    ]
    
    print(f"Input: {len(texts)} câu feedback")
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json={"texts": texts}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nTotal predictions: {result['total']}")
        
        for i, pred in enumerate(result['results'], 1):
            print(f"\n{i}. {pred['original_text']}")
            print(f"   → Sentiment: {pred['sentiment']['sentiment']}")
            print(f"   → Topic: {pred['topic']['topic']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_models_info():
    """Test models info endpoint"""
    print("\n" + "="*80)
    print("6. TEST MODELS INFO")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/models/info")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_error_cases():
    """Test error handling"""
    print("\n" + "="*80)
    print("7. TEST ERROR CASES")
    print("="*80)
    
    # Empty text
    print("\n--- Empty text ---")
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": ""}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Missing field
    print("\n--- Missing field ---")
    response = requests.post(
        f"{BASE_URL}/predict",
        json={}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def run_all_tests():
    """Chạy tất cả tests"""
    print("\n" + "="*80)
    print("🧪 TESTING VIETNAMESE FEEDBACK CLASSIFIER API")
    print("="*80)
    
    try:
        test_health()
        test_single_prediction()
        test_sentiment_only()
        test_topic_only()
        test_batch_prediction()
        test_models_info()
        test_error_cases()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED!")
        print("="*80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Không thể kết nối đến API")
        print("Hãy chắc chắn API đang chạy tại http://localhost:8000")
        print("Chạy: python main.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")

if __name__ == "__main__":
=======
# test_api.py - Script để test API
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("\n" + "="*80)
    print("1. TEST HEALTH CHECK")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_single_prediction():
    """Test single prediction"""
    print("\n" + "="*80)
    print("2. TEST SINGLE PREDICTION")
    print("="*80)
    
    test_cases = [
        "Giảng viên dạy rất hay và dễ hiểu",
        "Cơ sở vật chất trường còn thiếu thốn",
        "Chương trình đào tạo cần cập nhật thêm",
        "Thầy giảng bài không rõ ràng",
        "Phòng học sạch sẽ, thoáng mát"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n--- Test case {i} ---")
        print(f"Input: {text}")
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Sentiment: {result['sentiment']['sentiment']}", end="")
            if result['sentiment']['confidence']:
                print(f" (confidence: {result['sentiment']['confidence']:.2%})")
            else:
                print()
            print(f"Topic: {result['topic']['topic']}", end="")
            if result['topic']['confidence']:
                print(f" (confidence: {result['topic']['confidence']:.2%})")
            else:
                print()
        else:
            print(f"Error: {response.status_code} - {response.text}")

def test_sentiment_only():
    """Test sentiment-only prediction"""
    print("\n" + "="*80)
    print("3. TEST SENTIMENT ONLY")
    print("="*80)
    
    text = "Giảng viên nhiệt tình, tận tâm với sinh viên"
    print(f"Input: {text}")
    
    response = requests.post(
        f"{BASE_URL}/predict/sentiment",
        json={"text": text}
    )
    
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_topic_only():
    """Test topic-only prediction"""
    print("\n" + "="*80)
    print("4. TEST TOPIC ONLY")
    print("="*80)
    
    text = "Phòng thí nghiệm thiếu thiết bị"
    print(f"Input: {text}")
    
    response = requests.post(
        f"{BASE_URL}/predict/topic",
        json={"text": text}
    )
    
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "="*80)
    print("5. TEST BATCH PREDICTION")
    print("="*80)
    
    texts = [
        "Giảng viên dạy hay",
        "Trường cần đầu tư thêm máy tính",
        "Chương trình học quá khó",
        "Thư viện thiếu sách tham khảo"
    ]
    
    print(f"Input: {len(texts)} câu feedback")
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json={"texts": texts}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nTotal predictions: {result['total']}")
        
        for i, pred in enumerate(result['results'], 1):
            print(f"\n{i}. {pred['original_text']}")
            print(f"   → Sentiment: {pred['sentiment']['sentiment']}")
            print(f"   → Topic: {pred['topic']['topic']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_models_info():
    """Test models info endpoint"""
    print("\n" + "="*80)
    print("6. TEST MODELS INFO")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/models/info")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_error_cases():
    """Test error handling"""
    print("\n" + "="*80)
    print("7. TEST ERROR CASES")
    print("="*80)
    
    # Empty text
    print("\n--- Empty text ---")
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": ""}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Missing field
    print("\n--- Missing field ---")
    response = requests.post(
        f"{BASE_URL}/predict",
        json={}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def run_all_tests():
    """Chạy tất cả tests"""
    print("\n" + "="*80)
    print("🧪 TESTING VIETNAMESE FEEDBACK CLASSIFIER API")
    print("="*80)
    
    try:
        test_health()
        test_single_prediction()
        test_sentiment_only()
        test_topic_only()
        test_batch_prediction()
        test_models_info()
        test_error_cases()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED!")
        print("="*80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Không thể kết nối đến API")
        print("Hãy chắc chắn API đang chạy tại http://localhost:8000")
        print("Chạy: python main.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")

if __name__ == "__main__":
>>>>>>> ac82025111293788ba7c961553195313ed0bf602
    run_all_tests()