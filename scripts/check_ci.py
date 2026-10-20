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
                with urllib.request.urlopen(urllib.request.Request(jobs_url, headers={"User-Agent": "Mozilla/5.0"})) as jresp:
                    jdata = json.loads(jresp.read().decode("utf-8"))
                    print("\nJobs for run", runs[0]['id'])
                    for j in jdata.get("jobs", []):
                        print(f"  {j['name']}: {j['status']} - {j['conclusion']}")
            return runs[0] if runs else None
    except urllib.error.HTTPError as e:
        print(f"HTTPError {e.code}: {e.reason}")
        return None
    except Exception as e:
        print(f"Error checking CI: {e}")
        return None

if __name__ == "__main__":
    check()
