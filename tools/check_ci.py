import urllib.request
import json
import time

def check():
    url = "https://api.github.com/repos/pradeepkumar-ai-byte/VaxGuard/actions/runs"
    req = urllib.request.Request(url, headers={"User-Agent": "VaxGuard-CI-Checker"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            runs = data.get("workflow_runs", [])
            for r in runs[:5]:
                print(f"ID: {r['id']} | SHA: {r['head_sha'][:7]} | Event: {r['event']} | Status: {r['status']} | Conclusion: {r['conclusion']} | Title: {r['display_title']}")
            return runs[0] if runs else None
    except Exception as e:
        print(f"Error checking CI: {e}")
        return None

def check_jobs(run_id):
    url = f"https://api.github.com/repos/pradeepkumar-ai-byte/VaxGuard/actions/runs/{run_id}/jobs"
    req = urllib.request.Request(url, headers={"User-Agent": "VaxGuard-CI-Checker"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            jobs = data.get("jobs", [])
            for j in jobs:
                print(f"  Job: {j['name']} | Status: {j['status']} | Conclusion: {j['conclusion']}")
    except Exception as e:
        print(f"Error checking jobs: {e}")

if __name__ == "__main__":
    latest = check()
    if latest:
        print("\nJobs for latest run:")
        check_jobs(latest["id"])
