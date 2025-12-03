name: Weekly Report (Saturday 10:00)

on:
  schedule:
    - cron: "0 8 * * 6"    # 08:00 UTC = 10:00 România
  workflow_dispatch:

jobs:
  report:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Generate report
        run: |
          export PYTHONPATH="$PYTHONPATH:$(pwd)"
          python stockd/weekly_report.py

      - name: Commit report files
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "Weekly performance report"
          file_pattern: data/weekly_report/*
          commit_user_name: "StockD Agent"

      - name: Send Telegram notification
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: |
          TEXT="Weekly performance report is ready."
          curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d chat_id="${TELEGRAM_CHAT_ID}" \
            -d text="$TEXT"
