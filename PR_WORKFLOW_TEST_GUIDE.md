# Testing Pull Request Workflow on Master Branch

## ✅ Changes Made to `on_pull_request.yml`

Your PR workflow has been customized for master branch and root-level Python files:

### Updated:
1. **Trigger branches**: Now includes `master`, `main`, and `develop`
2. **Path filters**: Changed from `src/**` and `tests/**` to `*.py`
3. **Linting**: Updated to check root-level `.py` files
4. **Tests**: Simplified syntax check (no pytest needed)
5. **Code quality**: Updated pylint, mypy, bandit to work with root `.py` files
6. **Auto-assign**: Set to auto-assign PRs to `bkris2` (owner)

---

## 🧪 How to Test the PR Workflow

### **Option 1: Create a Test Branch & PR (Recommended)**

#### Step 1: Create a new branch
```powershell
git checkout -b test/pr-workflow
```

#### Step 2: Make a small change to trigger the workflow
```powershell
# Edit a Python file (e.g., add a comment)
echo "# Test PR workflow" >> Hello.py

# Or create a new Python file
echo "print('test')" > test_file.py
```

#### Step 3: Commit and push
```powershell
git add .
git commit -m "test: trigger PR workflow"
git push origin test/pr-workflow
```

#### Step 4: Create a Pull Request on GitHub
1. Go to: `https://github.com/bkris2/PythonAI`
2. Click **"Pull requests"** tab
3. Click **"New pull request"**
4. Set: **base: master** ← target, **compare: test/pr-workflow** ← your branch
5. Click **"Create pull request"**
6. Title: `test: PR workflow validation` (follows the convention)
7. Click **"Create pull request"**

#### Step 5: Watch the workflow run
- Workflow should start automatically (check **Actions** tab)
- Watch logs in real-time

---

### **Option 2: Test Locally with `act` (Faster)**

If you want to test without pushing:

```powershell
# Install act (one-time)
# From: https://github.com/nektos/act

# Run the PR workflow locally
act pull_request

# Or run with specific event
act pull_request -e event.json
```

---

## 📋 What the PR Workflow Tests

When you create a PR to `master`, it will:

1. ✅ **Validate PR title** – Must follow convention (e.g., `feat:`, `fix:`, `test:`)
2. ✅ **Check file sizes** – No files >10MB
3. ✅ **Test on Python 3.9, 3.10, 3.11** (matrix testing)
4. ✅ **Flake8 linting** – Code style check
5. ✅ **Black formatting** – Code format validation
6. ✅ **Python syntax** – Compile check
7. ✅ **Pylint** – Code analysis
8. ✅ **Mypy** – Type checking
9. ✅ **Bandit** – Security scanning
10. ✅ **Auto-comment** – Posts results on PR
11. ✅ **Auto-assign** – Assigns PR to owner

---

## 🎯 Expected Outcomes

### **Successful PR Workflow Run:**
- All checks pass (green checkmarks)
- Automated comment posted: "✅ All checks passed! Ready for merge."
- PR auto-assigned to `bkris2`

### **If Workflow Fails:**
- Red X on PR
- Automated comment: "❌ Some checks failed. Please review the logs above."
- See detailed error logs in Actions tab

---

## 📝 Example PR Titles (Valid/Invalid)

### ✅ Valid (Will pass validation)
- `feat: add SageMaker integration`
- `fix: correct linear regression bug`
- `test: add unit tests for migration`
- `docs: update migration guide`
- `refactor: improve code structure`

### ❌ Invalid (Will fail validation)
- `Update code` ← Missing prefix
- `Fix bug` ← Missing prefix and colon
- `random title` ← Not following convention

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Workflow doesn't run | Check PR is to `master` branch |
| Linting fails | Run `flake8 *.py` locally to see issues |
| Black format fails | Run `black *.py` to auto-format |
| Type checking fails | Run `mypy *.py` to see type errors |
| Bandit security warnings | Review security issues in `bandit *.py` output |

---

## 📂 Files Modified

- ✅ `.github/workflows/on_pull_request.yml` – Updated for master branch
- 📄 This guide – Quick testing reference

---

## ✨ Next Steps

```powershell
# 1. Commit workflow changes
git add .github/workflows/on_pull_request.yml
git commit -m "ci: customize PR workflow for master branch"

# 2. Push to master
git push origin master

# 3. Create a test PR (follow steps above in Option 1)
```

---

**Status:** ✅ PR workflow is ready to test!

Create a test branch and PR to trigger it.

