# Push vs Pull Request Workflows – Comparison & Examples

This document explains the differences between **push workflows** and **pull request workflows** with practical examples.

---

## Quick Comparison Table

| Aspect | **Push Workflow** | **Pull Request Workflow** |
|--------|------------------|-------------------------|
| **Trigger** | Commits pushed to specified branches | PR opened, updated, or reopened |
| **Use Case** | CI/CD: build, test, deploy | Code review: validate changes before merge |
| **Access to Secrets** | Full access (risky for forks) | Limited for security |
| **Commit Context** | Direct access to branch | Merge commit context |
| **Typical Actions** | Test, build, deploy, release | Validate, lint, comment on PR |
| **When to Run** | After code lands on branch | Before code lands on branch |

---

## Workflow Triggers

### **Push Workflow (`on: push`)**
```yaml
on:
  push:
    branches: [main, develop, staging]
    paths:
      - 'src/**'
      - 'requirements.txt'
```

**Key Points:**
- Runs after commits are pushed to the branch
- Has direct access to repository secrets (use with caution)
- Can deploy directly to production
- Ideal for release automation

### **Pull Request Workflow (`on: pull_request`)**
```yaml
on:
  pull_request:
    branches: [main, develop]
    types: [opened, synchronize, reopened, ready_for_review]
    paths:
      - 'src/**'
      - 'tests/**'
```

**Key Points:**
- Runs on PR events (open, update, reopen)
- Limited secret access for security (especially for forks)
- Cannot automatically deploy without approval
- Ideal for validation and code review checks

---

## Real-World Workflow Examples

### **on_push.yml** – What It Does

1. **test-and-build** job:
   - Checks out code
   - Installs dependencies
   - Runs linting (flake8)
   - Runs tests (pytest)
   - Builds package distribution

2. **deploy-to-staging** job (conditional):
   - Runs only if push is to `develop` branch
   - Deploys code to staging environment

3. **notification** job:
   - Always runs as final step
   - Sends deployment notifications

**Ideal for:**
- ✅ Running comprehensive tests after code lands
- ✅ Building release artifacts
- ✅ Deploying to staging/production automatically
- ✅ Publishing packages

---

### **on_pull_request.yml** – What It Does

1. **validate-pr** job:
   - Validates PR title format (feat:, fix:, etc.)
   - Checks for large files (>10MB)

2. **test-changes** job (matrix strategy):
   - Tests on Python 3.9, 3.10, 3.11 simultaneously
   - Runs linting, formatting checks, pytest
   - Uploads code coverage reports

3. **code-quality** job:
   - Static analysis with pylint
   - Type checking with mypy
   - Security scan with bandit

4. **pr-comment** job:
   - Posts automated comment on PR with results

5. **auto-assign** job:
   - Auto-assigns PR to reviewers for new contributors

**Ideal for:**
- ✅ Catching issues before code lands on main branch
- ✅ Enforcing code standards
- ✅ Multi-version testing in parallel
- ✅ Automating code review feedback

---

## Key Differences Explained

### **1. Permissions & Secrets**

**Push Workflow:**
```yaml
# Full access to secrets (dangerous for forks!)
env:
  DATABASE_PASSWORD: ${{ secrets.DB_PASSWORD }}
  API_KEY: ${{ secrets.PROD_API_KEY }}
```

**Pull Request Workflow:**
```yaml
# Limited secret access for security
permissions:
  contents: read
  pull-requests: write
  checks: write
```

### **2. Conditional Deployments**

**Push Workflow:**
```yaml
deploy-to-prod:
  if: github.ref == 'refs/heads/main'
  needs: test-and-build
  steps:
    - run: ./deploy-prod.sh
```

**Pull Request Workflow:**
```yaml
# Cannot deploy directly, only validate
validate-deployment:
  if: contains(github.event.pull_request.labels.*.name, 'ready-to-deploy')
  steps:
    - run: echo "This PR is ready for deployment (manual required)"
```

### **3. Event Context**

**Push - Direct Commit:**
```yaml
steps:
  - run: echo "Commit: ${{ github.sha }}"
         # Example output: ab1234cd567...
```

**Pull Request - Merge Commit:**
```yaml
steps:
  - run: echo "PR Head: ${{ github.event.pull_request.head.sha }}"
         echo "PR Base: ${{ github.event.pull_request.base.sha }}"
```

---

## When to Use Which?

### **Use Push Workflow When:**
- 🚀 Deploying to production/staging
- 📦 Building and publishing packages (PyPI, npm, etc.)
- 🏷️ Creating releases or tags
- 📊 Running expensive long-running tasks
- 🔄 Syncing across multiple repositories

### **Use Pull Request Workflow When:**
- ✅ Validating code before merge
- 🔍 Running code quality checks
- 📝 Enforcing coding standards
- 🧪 Running full test suite on multiple Python versions
- 💬 Adding automated feedback on PR

---

## Workflow Execution Order

### **Push Workflow Timeline**
```
Developer Commits → Pushes to branch
        ↓
[on: push] triggered
        ↓
Jobs run in parallel (test, lint, build)
        ↓
Deploy job runs (if main branch)
        ↓
Notification sent
```

### **Pull Request Workflow Timeline**
```
Developer Creates/Updates PR
        ↓
[on: pull_request] triggered
        ↓
Validation, Testing, Code Quality jobs run in parallel
        ↓
Automated comment posted on PR
        ↓
Reviewers notified
        ↓
Developer Merges → [on: push] triggered
```

---

## Security Best Practices

### **For Push Workflows:**
- ⚠️ **Limit secret exposure**: Only pass secrets to necessary steps
- ⚠️ **Use environments**: Set different approval rules for prod vs. staging
- ⚠️ **Restrict branch protection**: Require reviews before merge

```yaml
deploy-prod:
  environment:
    name: production
    url: https://prod.example.com
  if: github.ref == 'refs/heads/main'
  needs: test-and-build
```

### **For Pull Request Workflows:**
- ✅ **Limited secret access**: Forks don't get secrets by default
- ✅ **Code review required**: Tests run before merge
- ✅ **Safe validation**: Can test without deploying

---

## Common Patterns

### **Pattern 1: Test on PR, Deploy on Push**
```yaml
# on_pull_request.yml
on: [pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: pytest

# on_push.yml
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: pytest
  deploy:
    needs: test
    steps:
      - run: ./deploy.sh
```

### **Pattern 2: Strict PR Validation + Auto-deployment**
```yaml
# on_pull_request.yml
jobs:
  validate:
    steps:
      - run: pytest -v
      - run: flake8 .
      - run: mypy .

# on_push.yml
jobs:
  deploy:
    if: github.ref == 'refs/heads/main'
    steps:
      - run: docker push gcr.io/...
```

---

## Files in This Folder

- `on_push.yml` – Complete push workflow example
- `on_pull_request.yml` – Complete pull request workflow example
- `01_python_ci_starter.yml` – Basic starter template
- `02_manual_deploy_starter.yml` – Manual deployment template
- `03_scheduled_task_starter.yml` – Cron-based tasks

---

## Quick Reference

**Start with push workflow when debugging:**
```bash
# Test locally with act
act push
```

**Test PR workflow locally:**
```bash
act pull_request
```

See `GITHUB_WORKFLOWS_GUIDE.md` for full documentation and troubleshooting.

---

**Last Updated:** February 2026
