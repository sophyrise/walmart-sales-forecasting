# Project Setup Instructions

## Prerequisites
- Python 3.10+
- Git
- Kaggle API key

## For New Contributors

1. Clone repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/walmart-sales-forecasting.git
   cd walmart-sales-forecasting
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

6. Create your working branch:
   ```bash
   git checkout -b setup/YOUR_NAME
   ```

## DagsHub Integration

MLflow tracking automatically syncs to DagsHub at:
https://dagshub.com/sophyrise/walmart-sales-forecasting

## Branch Strategy

- `main`: Production-ready code
- `setup/*`: Environment and infrastructure setup
- `feature/*`: Feature branches (notebooks, models)

Always create a new branch for your work:
```bash
git checkout -b feature/model_xgboost
```

## Working with Teammate

Pull latest changes before starting work:
```bash
git checkout main
git pull origin main
git checkout YOUR_BRANCH
```

Push your changes regularly:
```bash
git add .
git commit -m "your message"
git push origin YOUR_BRANCH
```
