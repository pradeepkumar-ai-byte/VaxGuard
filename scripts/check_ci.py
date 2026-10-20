import urllib.request
import json

def check():
    url = "https://api.github.com/repos/pradeepkumar-ai-byte/VaxGuard/actions/runs?per_page=3"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            runs = data.get("workflow_runs", [])
            for r in runs[:3]:
                print(f"ID: {r['id']} | SHA: {r['head_sha'][:7]} | Status: {r['status']} | Conclusion: {r['conclusion']} | Title: {r['display_title'][:50]}")
            if runs:
                jobs_url = f"https://api.github.com/repos/pradeepkumar-ai-byte/VaxGuard/actions/runs/{runs[0]['id']}/jobs"
                j_req = urllib.request.Request(jobs_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(j_req) as j_resp:
                    j_data = json.loads(j_resp.read().decode("utf-8"))
                    for j in j_data.get('jobs', []):
                        failed_step = next((s['name'] for s in j.get('steps', []) if s.get('conclusion') == 'failure'), None)
                        print(f"  --> Job: {j['name']} | ID: {j['id']} | Failed Step: {failed_step} | Conclusion: {j['conclusion']}")
            return runs[0] if runs else None
    except urllib.error.HTTPError as e:
        reset_ts = e.headers.get("x-ratelimit-reset")
        if reset_ts:
            import datetime
            reset_dt = datetime.datetime.fromtimestamp(int(reset_ts))
            print(f"HTTPError {e.code}: rate limited until {reset_dt}")
        else:
            print(f"HTTPError {e.code}: {e.reason}")
        return None
    except Exception as e:
        print(f"Error checking CI: {e}")
        return None

if __name__ == "__main__":
    check()
