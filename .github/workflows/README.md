# GitHub Workflows – Starter Templates

This folder contains example GitHub Actions workflows to help you get started with CI/CD automation.

## Templates

### 1. **01_python_ci_starter.yml**
Basic Python test runner that:
- Tests on Python 3.9, 3.10, 3.11 (matrix)
- Checks code with flake8
- Runs pytest

**Triggers:** Push to main/develop, Pull Requests

**Usage:** Copy to `.github/workflows/ci.yml` and update `requirements.txt` path if needed.

---

### 2. **02_manual_deploy_starter.yml**
Manual deployment workflow with inputs:
- Choose environment (dev, staging, prod)
- Optional version input
- Placeholder for your deployment script

**Triggers:** Manual (via `workflow_dispatch`)

**Usage:** 
1. Copy to `.github/workflows/deploy.yml`
2. Go to Actions → find workflow → "Run workflow"
3. Select environment and optional version

---

### 3. **03_scheduled_task_starter.yml**
Runs a task on schedule:
- Weekly on Sunday at 2 AM UTC (editable cron)
- Configurable Python script
- Optional git commit of logs

**Triggers:** Scheduled (cron) + Manual

**Usage:** Update `scripts/scheduled_task.py` path to your script.

---

## Quick Start

1. **Choose a template** from above
2. **Copy to** `.github/workflows/your-name.yml`
3. **Edit** to match your repo structure
4. **Commit & push** to trigger the workflow
5. **Monitor** in GitHub Actions tab

---

## Common Edits

**Change Python versions:**
```yaml
matrix:
  python-version: ['3.10', '3.11', '3.12']
```

**Change trigger branches:**
```yaml
on:
  push:
    branches: [main, staging]
```

**Change cron schedule:**
```yaml
schedule:
  - cron: '0 0 * * *'  # Daily at midnight UTC
```

**Add secrets:**
```yaml
env:
  API_KEY: ${{ secrets.MY_API_KEY }}
```

---

## Learn More

See `GITHUB_WORKFLOWS_GUIDE.md` in the parent directory for full documentation, advanced examples, and troubleshooting.

---

**Last Updated:** February 2026
