import sys
print(sys.path)  # Debug: check module paths

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=8000)
