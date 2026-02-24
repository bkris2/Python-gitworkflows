# GitHub Workflows Learning Guide

A comprehensive guide to setting up, writing, and troubleshooting GitHub Actions workflows.

## Table of Contents
1. [Basics](#basics)
2. [Workflow Structure](#workflow-structure)
3. [Common Triggers](#common-triggers)
4. [Example Workflows](#example-workflows)
5. [Advanced Topics](#advanced-topics)
6. [Troubleshooting](#troubleshooting)

---

## Basics

### What is GitHub Actions?
GitHub Actions is a CI/CD platform that lets you automate build, test, and deployment tasks directly in your repository.

### Key Concepts
- **Workflow**: A YAML file defining automation jobs and steps.
- **Event**: What triggers the workflow (e.g., push, pull_request, schedule).
- **Job**: A set of steps that execute on the same runner.
- **Step**: Individual tasks within a job (run command, use action).
- **Action**: Reusable units of code (e.g., `actions/checkout@v3`).
- **Runner**: A machine that executes workflows (GitHub-hosted or self-hosted).

---

## Workflow Structure

### Basic YAML Template

```yaml
name: Workflow Name

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  job-name:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Run a command
        run: echo "Hello, World!"
```

### File Location
All workflows must be placed in `.github/workflows/` directory as YAML files.

---

## Common Triggers

### 1. **Push to branches**
```yaml
on:
  push:
    branches: [main, develop]
    paths: ['src/**', '*.py']
```

### 2. **Pull Request**
```yaml
on:
  pull_request:
    branches: [main]
    types: [opened, synchronize, reopened]
```

### 3. **Schedule (Cron)**
```yaml
on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight UTC
```

### 4. **Manual Trigger (Workflow Dispatch)**
```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy'
        required: true
        default: 'staging'
```

### 5. **Release**
```yaml
on:
  release:
    types: [published, created]
```

### 6. **Multiple Events**
```yaml
on: [push, pull_request, workflow_dispatch]
```

---

## Example Workflows

### Example 1: Python Testing & Linting

```yaml
name: Python CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache pip packages
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest flake8

      - name: Lint with flake8
        run: |
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
          flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

      - name: Run tests with pytest
        run: pytest tests/ -v --cov=src

      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

### Example 2: Build & Deploy to GCP Vertex AI

```yaml
name: Deploy to Vertex AI

on:
  push:
    branches: [main]
    paths: ['src/**', 'model/**', '.github/workflows/deploy.yml']

env:
  PROJECT_ID: my-gcp-project
  REGION: us-central1
  GCR_REPO: gcr.io/my-gcp-project

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    permissions:
      contents: 'read'
      id-token: 'write'

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          workload_identity_provider: ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
          service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v1

      - name: Build Docker image
        run: |
          docker build -t ${{ env.GCR_REPO }}/model:${{ github.sha }} .
          docker tag ${{ env.GCR_REPO }}/model:${{ github.sha }} ${{ env.GCR_REPO }}/model:latest

      - name: Push to Google Container Registry
        run: |
          gcloud auth configure-docker
          docker push ${{ env.GCR_REPO }}/model:${{ github.sha }}
          docker push ${{ env.GCR_REPO }}/model:latest

      - name: Deploy to Vertex AI
        run: |
          gcloud ai models upload \
            --region=${{ env.REGION }} \
            --display-name="model-${{ github.sha }}" \
            --container-image-uri="${{ env.GCR_REPO }}/model:${{ github.sha }}"
```

### Example 3: Automated Version Bumping & Release

```yaml
name: Version Bump and Release

on:
  push:
    branches: [main]

jobs:
  bump-version:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Bump version and create tag
        uses: anothrNick/github-tag-action@1.64.0
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          DEFAULT_BUMP: patch
          RELEASE_BRANCHES: main

      - name: Get latest tag
        id: get_tag
        run: |
          TAG=$(git describe --tags --abbrev=0)
          echo "version=${TAG#v}" >> $GITHUB_OUTPUT

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          tag_name: ${{ steps.get_tag.outputs.version }}
          body: "Release ${{ steps.get_tag.outputs.version }}"
          draft: false
          prerelease: false
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### Example 4: Data Validation on PR & Push

```yaml
name: Data Validation

on:
  push:
    branches: [main, develop]
    paths: ['data/**']
  pull_request:
    paths: ['data/**']

jobs:
  validate-data:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install pandas great-expectations

      - name: Validate data with Great Expectations
        run: |
          python scripts/validate_data.py

      - name: Check for sensitive data
        run: |
          pip install detect-secrets
          detect-secrets scan --baseline .secrets.baseline
```

### Example 5: Scheduled Model Retraining

```yaml
name: Scheduled Model Retraining

on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM UTC
  workflow_dispatch:

env:
  GCP_PROJECT: my-gcp-project
  REGION: us-central1

jobs:
  retrain-model:
    runs-on: ubuntu-latest

    permissions:
      contents: 'read'
      id-token: 'write'

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          workload_identity_provider: ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
          service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v1

      - name: Trigger training pipeline
        run: |
          gcloud ai custom-jobs create \
            --region=${{ env.REGION }} \
            --display-name="weekly-retrain-${{ github.run_number }}" \
            --python-package-gcs-uri=gs://my-bucket/training.tar.gz \
            --python-module-name=trainer.train \
            --machine-type=n1-standard-4 \
            --replica-count=1

      - name: Send notification
        if: failure()
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '⚠️ Scheduled retraining failed!'
            })
```

---

## Advanced Topics

### 1. **Secrets Management**
Store sensitive data (API keys, credentials) securely.

```yaml
steps:
  - name: Use secret
    run: echo ${{ secrets.DATABASE_PASSWORD }}
    env:
      DB_PASS: ${{ secrets.DATABASE_PASSWORD }}
```

**Set in GitHub:**
1. Go to Repo → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `DATABASE_PASSWORD`, Value: `your-password`

### 2. **Conditional Steps**

```yaml
steps:
  - name: Deploy to production
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    run: ./deploy.sh
```

### 3. **Matrix Strategy (Test Multiple Versions)**

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: [3.8, 3.9, '3.10']
  fail-fast: false

runs-on: ${{ matrix.os }}
```

### 4. **Artifacts & Caching**

```yaml
- name: Upload test results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: test-results/

- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: ~/.m2/repository
    key: ${{ runner.os }}-maven-${{ hashFiles('**/pom.xml') }}
```

### 5. **Job Dependencies**

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - run: ./deploy.sh
```

### 6. **Reusable Workflows**

Create a reusable workflow in `.github/workflows/reusable.yml`:

```yaml
name: Reusable Test Workflow

on:
  workflow_call:
    inputs:
      python-version:
        required: false
        type: string
        default: '3.10'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ inputs.python-version }}
      - run: pytest
```

Call it from another workflow:

```yaml
jobs:
  call-test:
    uses: ./.github/workflows/reusable.yml
    with:
      python-version: '3.11'
```

---

## Troubleshooting

### 1. **Workflow Won't Trigger**
- Check branch/path filters match your push
- Verify file is in `.github/workflows/`
- Check YAML syntax (use `yaml lint` online)
- Try `workflow_dispatch` manually to debug

### 2. **Step Fails Silently**
- Add `set -x` to bash scripts to debug
- Check logs in Actions tab
- Use `if: always()` to run follow-up steps

### 3. **Secrets Not Working**
- Verify secret name is correct (case-sensitive)
- Check permissions in Settings → Actions
- Use `${{ secrets.SECRET_NAME }}` syntax

### 4. **Slow Builds**
- Use caching for dependencies
- Use matrix to parallelize
- Use `fail-fast: true` to stop on first failure
- Consider self-hosted runners

### 5. **Out of Disk Space**
Clean up in workflow:
```yaml
- name: Free disk space
  run: |
    sudo rm -rf /usr/local/lib/android
    sudo rm -rf /usr/share/dotnet
    df -h
```

### 6. **Debugging Commands**
```yaml
- name: Debug info
  run: |
    echo "Branch: ${{ github.ref }}"
    echo "Event: ${{ github.event_name }}"
    echo "Actor: ${{ github.actor }}"
    echo "Workspace: ${{ github.workspace }}"
```

---

## Best Practices

- ✅ Use `actions/checkout@v3` to get your code
- ✅ Cache dependencies to speed up jobs
- ✅ Use matrix strategy for multiple versions
- ✅ Set meaningful job names and step descriptions
- ✅ Use conditionals to avoid unnecessary runs
- ✅ Secure secrets in GitHub repo settings
- ✅ Test workflows locally first with `act`
- ✅ Document your workflows in README
- ✅ Use reusable workflows for DRY principle
- ✅ Monitor workflow costs and duration

---

## Useful Actions & Marketplace

- [`actions/checkout`](https://github.com/actions/checkout) – Clone repo
- [`actions/setup-python`](https://github.com/actions/setup-python) – Setup Python
- [`actions/upload-artifact`](https://github.com/actions/upload-artifact) – Store outputs
- [`codecov/codecov-action`](https://github.com/codecov/codecov-action) – Code coverage
- [`google-github-actions/auth`](https://github.com/google-github-actions/auth) – GCP auth
- [`softprops/action-gh-release`](https://github.com/softprops/action-gh-release) – Create releases

Browse more at [GitHub Marketplace](https://github.com/marketplace?type=actions)

---

## Testing Workflows Locally

Use `act` tool to test workflows without pushing:

```bash
# Install act
# https://github.com/nektos/act

# Run workflow
act -j test

# Run specific workflow
act -W .github/workflows/test.yml
```

---

**Last Updated:** February 2026
**Version:** 1.0
