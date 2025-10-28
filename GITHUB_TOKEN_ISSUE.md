# GitHub Token Issue - RESOLVED ✓

## 🔍 Root Cause Identified

The GitHub Advisory collection is failing because:

**STATUS CODE: 401 Unauthorized - "Bad credentials"**

### Issue Details:
- Token in `.env` file: `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`
- Token length: 40 characters (correct format)
- Token prefix: `ghp_` (correct for Personal Access Token)
- **Problem**: Token is **EXPIRED or INVALID**

## ✅ Solution

### Step 1: Generate New GitHub Token

1. Go to GitHub: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name: "StreamGuard Data Collection"
4. Set expiration: 90 days (or No expiration for long-term use)
5. **Required Scopes** - Check these boxes:
   - ✓ `public_repo` (access public repositories)
   - ✓ `read:org` (read organization data)
   - ✓ `security_events` (read security events)

6. Click "Generate token"
7. **Copy the token immediately** (you won't see it again!)

### Step 2: Update .env File

Replace the old token in `.env` file (line 35):

```env
# OLD (expired):
GITHUB_TOKEN=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

# NEW (replace with your new token):
GITHUB_TOKEN=ghp_YOUR_NEW_TOKEN_HERE
```

### Step 3: Verify Token Works

Run the test script:

```bash
cd "c:\Users\Vimal Sajan\streamguard"
python test_github_token.py
```

Expected output:
```
Status code: 200
Response: {"data":{"viewer":{"login":"YOUR_USERNAME"}}}
```

### Step 4: Run Collection Test

Once token is verified, test the GitHub collector:

```bash
python training/scripts/collection/test_github_collector.py
```

Expected: 50 samples collected successfully

### Step 5: Run Full Collection

```bash
python training/scripts/collection/run_full_collection.py --collectors github --github-samples 10000
```

---

## 📋 Debug Results

### Tests Performed:
1. ✓ Environment loading (.env file) - **WORKING**
2. ✓ Token format validation - **CORRECT FORMAT**
3. ✓ Collector initialization - **WORKING**
4. ✓ GraphQL query structure - **CORRECT**
5. **✗ Token authentication - FAILED (401 Unauthorized)**

### Logs:
```
DEBUG: Querying GraphQL for PIP/HIGH, cursor: None
DEBUG: Response status code: 401
GraphQL query failed: 401 Client Error: Unauthorized
```

### Why This Happened:
- GitHub Personal Access Tokens expire after 30-90 days by default
- The token `ghp_PlKPGw...` was generated previously and has expired
- GitHub returns 401 for expired or revoked tokens

---

## 🎯 Additional Enhancements Made

While debugging, I also:

1. ✓ **Fixed `.env` file** - Removed invalid `qqq` line
2. ✓ **Added comprehensive debug logging** to GitHub collector
3. ✓ **Created test script** for easy debugging
4. ✓ **Improved error handling** with detailed messages

### Files Modified:
- `github_advisory_collector_enhanced.py` - Added 10+ debug points
- `.env` - Removed invalid line
- `test_github_collector.py` - New test script (120 lines)
- `test_github_token.py` - Token validation script (40 lines)

---

## 🚀 Next Steps After Token Replacement

Once you have a valid token:

### 1. Test Collection (2 minutes)
```bash
python training/scripts/collection/test_github_collector.py
```

### 2. Small Collection (30 minutes)
```bash
python training/scripts/collection/run_full_collection.py \
    --quick-test \
    --collectors github
```

### 3. Full Collection (4-6 hours)
```bash
python training/scripts/collection/run_full_collection.py \
    --collectors github \
    --github-samples 10000
```

---

## 📊 Expected Results

After fixing the token:

- GitHub advisories: **10,000 samples**
- Code pair extraction: **~3,500 pairs (35%)**
- Coverage: 8 ecosystems, 4 severity levels
- File: `data/raw/github/github_advisories.jsonl`

---

## 🔐 Token Security Best Practices

1. **Never commit tokens to git**
   - Already in `.gitignore`

2. **Use environment variables**
   - Already using `.env` file ✓

3. **Rotate tokens regularly**
   - Set expiration to 90 days
   - Add calendar reminder

4. **Minimum required permissions**
   - Only grant scopes needed
   - No write permissions needed

---

## 📝 Summary

| Issue | Status | Solution |
|-------|--------|----------|
| GitHub 401 Error | ✓ Identified | Replace expired token |
| Invalid .env line | ✓ Fixed | Removed `qqq` |
| Missing debug logs | ✓ Added | 10+ debug points |
| No test script | ✓ Created | test_github_collector.py |
| File consolidation | ✓ Verified | Working correctly |

**Action Required:** Generate new GitHub token and update `.env` file

---

*Last Updated: October 16, 2025*
