# run.py - Script để chạy nhanh app
import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def check_requirements():
    """Kiểm tra các yêu cầu cần thiết"""
    print("🔍 Checking requirements...")
    
    errors = []
    
    # Check models
    # models_dir = Path("tuned_models")
    # Dòng mới - luôn tìm đúng vị trí dù chạy từ đâu
    models_dir = Path(__file__).parent / "tuned_models"
    if not models_dir.exists():
        errors.append("❌ Thư mục tuned_models không tồn tại")
    
    # # Check index.html
    # if not Path("static/index.html").exists():
    #     errors.append("❌ File static/index.html không tồn tại")
    
    # # Check main.py
    # if not Path("main.py").exists():
    #     errors.append("❌ File main.py không tồn tại")

    # Check index.html
    base_dir = Path(__file__).parent
    if not (base_dir / "static/index.html").exists():
        errors.append("❌ File static/index.html không tồn tại")

    # Check main.py
    if not (base_dir / "main.py").exists():
        errors.append("❌ File main.py không tồn tại")
    
    return errors

def main():
    base_dir = Path(__file__).parent
    print("=" * 80)
    print("🚀 VIETNAMESE FEEDBACK CLASSIFIER - QUICK START")
    print("=" * 80)
    
    # Check requirements
    errors = check_requirements()
    
    if errors:
        print("\n⚠️  Phát hiện lỗi:")
        for error in errors:
            print(f"   {error}")
        print("\n📝 Hướng dẫn fix:")
        print("   1. Chạy: python setup.py")
        print("   2. Đảm bảo đã có models từ bước training")
        print("   3. Đảm bảo đã có file index.html trong thư mục static/")
        sys.exit(1)
    
    print("\n✓ Tất cả yêu cầu đã được đáp ứng!")
    print("\n" + "=" * 80)
    print("🌐 STARTING SERVER...")
    print("=" * 80)
    print("\n📍 Server sẽ chạy tại:")
    print("   - Frontend: http://localhost:8000")
    print("   - API Docs: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print("\n⌨️  Nhấn Ctrl+C để dừng server")
    print("=" * 80 + "\n")
    
    # # Wait a bit then open browser
    # def open_browser():
    #     time.sleep(2)
    #     print("\n🌐 Đang mở trình duyệt...")
    #     webbrowser.open('http://localhost:8000')
    
    # import threading
    # browser_thread = threading.Thread(target=open_browser)
    # browser_thread.daemon = True
    # browser_thread.start()
    
    # Run the server
    try:
        # subprocess.run([sys.executable, "main.py"])
        subprocess.run([sys.executable, str(base_dir / "main.py")])
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("👋 Server đã dừng!")
        print("=" * 80)
        sys.exit(0)

if __name__ == "__main__":
    main()