.PHONY: web web-backend web-frontend install clean

web:
	@./scripts/start_web.sh

web-backend:
	@echo "Starting backend API on port 5000..."
	@cd web2_api && flask run --port 5000

web-frontend:
	@echo "Starting frontend on port 5173..."
	@cd web2 && npm run dev

install:
	@echo "Installing backend dependencies..."
	@pip install -e .
	@pip install flask flask-cors cryptography
	@echo "Installing frontend dependencies..."
	@cd web2 && npm install

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf dist build .pytest_cache 2>/dev/null || true