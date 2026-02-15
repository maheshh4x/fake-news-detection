import csv
import os
from datetime import datetime

def save_feedback(feedback_file, text, verdict, user_feedback):
    file_exists = os.path.exists(feedback_file)

    with open(feedback_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header only first time
        if not file_exists:
            writer.writerow(["time", "text", "final_verdict", "user_feedback"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            text[:300],  # limit long text
            verdict,
            user_feedback
        ])
