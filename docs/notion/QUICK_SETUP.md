# StreamGuard Notion - Quick Setup Guide

**Time Required:** 10-15 minutes
**Difficulty:** Easy

This is the fastest way to get your StreamGuard Notion workspace set up and running.

---

## Prerequisites

- Notion account (free or paid)
- Access to the files in this directory:
  - `NOTION_STRUCTURE.md` - Detailed setup instructions
  - `NOTION_CONTENT.md` - All page content
  - `issues.csv` - Issues database data
  - `tasks.csv` - Tasks database data

---

## 5-Step Quick Setup

### Step 1: Create Main Workspace (2 minutes)

1. Open Notion
2. Click "+ New page" in your sidebar
3. Name it **"StreamGuard Project"**
4. Add icon: Click the icon area → Search for "shield" → Select 🛡️
5. Add cover (optional): Click "Add cover" → Choose a color or image

---

### Step 2: Import Issues Database (3 minutes)

1. In your StreamGuard Project page, type `/table`
2. Select "Table - Inline"
3. Click the `⋯` menu → Select "Merge with CSV"
4. Upload `issues.csv` from this directory
5. Rename the database to **"Issues Tracker"**

**Configure the database:**
- Click on "Status" column → Change type to "Select"
- Add color coding:
  - Critical → Red
  - High → Orange
  - Medium → Yellow
  - Low → Gray

---

### Step 3: Import Tasks Database (3 minutes)

1. Below the Issues Tracker, type `/table`
2. Select "Table - Inline"
3. Click the `⋯` menu → Select "Merge with CSV"
4. Upload `tasks.csv` from this directory
5. Rename the database to **"Tasks"**

**Configure the database:**
- Click on "Status" column → Change type to "Select"
- Add color coding:
  - Completed → Green
  - In Progress → Blue
  - Not Started → Gray
  - Blocked → Purple

---

### Step 4: Create Essential Pages (5 minutes)

**Create these pages below your databases:**

1. **Dashboard**
   - Type `/page` → Name it "Dashboard"
   - Copy content from `NOTION_CONTENT.md` → "DASHBOARD PAGE" section
   - Paste into your new page

2. **Project Progress**
   - Type `/page` → Name it "Project Progress"
   - Copy content from `NOTION_CONTENT.md` → "PROJECT PROGRESS PAGE" section
   - Paste into your new page

3. **Quick Reference**
   - Type `/page` → Name it "Quick Reference Commands"
   - Copy content from `NOTION_CONTENT.md` → "QUICK REFERENCE COMMANDS PAGE" section
   - Paste into your new page

---

### Step 5: Link Everything (2 minutes)

1. Go to your Tasks database
2. Add a new property: Click "+" → "Relation"
3. Name it "Related Issue"
4. Select "Issues Tracker" as the related database
5. Enable "Show on Issues Tracker" for two-way linking

**Done!** You now have a functional StreamGuard tracking workspace.

---

## Quick Verification Checklist

After setup, verify you have:

- ✅ Main "StreamGuard Project" page with icon
- ✅ Issues Tracker database with 7 issues imported
- ✅ Tasks database with 37 tasks imported
- ✅ Dashboard page with quick stats
- ✅ Project Progress page with phase details
- ✅ Quick Reference page with commands
- ✅ Two-way relation between Tasks and Issues

---

## Next Steps (Optional Enhancements)

### Add More Views

**For Issues Tracker:**
1. Click "New view" → Board
2. Group by "Status"
3. Name it "Active Issues"

**For Tasks:**
1. Click "New view" → Board
2. Group by "Status"
3. Filter: Status is not "Completed"
4. Name it "Active Tasks"

### Create Sub-pages

Create pages for each phase under "Project Progress":
1. Phase 1: CVE Collection
2. Phase 2: GitHub Advisories
3. Phase 3: Repository Mining
4. Phase 4: Synthetic Generation
5. Phase 5: Master Orchestrator
6. Phase 6: Model Training

(Copy content from the PROJECT PROGRESS PAGE section in `NOTION_CONTENT.md`)

### Customize Appearance

- Add covers to each page
- Use callout blocks for important notes
- Add dividers for better section separation
- Customize property icons in databases

---

## Troubleshooting

### CSV Import Issues

**Problem:** CSV won't import or data looks wrong

**Solutions:**
1. Ensure CSV file is UTF-8 encoded
2. Open CSV in Notepad, save as UTF-8
3. Try importing from Notion desktop app instead of web

**Problem:** Dates not formatting correctly

**Solution:**
1. After import, click on Date column
2. Change date format to your preference
3. Manually adjust any incorrect dates

### Database Relations

**Problem:** Can't create relation between Tasks and Issues

**Solution:**
1. Ensure both databases exist first
2. Open Tasks database → Add property → Relation
3. Select "Issues Tracker"
4. Enable two-way linking

---

## Usage Tips

### Daily Workflow

**Morning:**
1. Open Dashboard
2. Review "Active Tasks"
3. Update task statuses as you work

**Evening:**
1. Mark completed tasks as done
2. Add any new issues discovered
3. Update task progress

### Weekly Review

**Every Friday:**
1. Review Issues Tracker
2. Close resolved issues
3. Update project progress percentages
4. Plan next week's tasks

---

## Advanced Features (For Later)

Once you're comfortable with the basics:

1. **Linked Databases** - Show filtered views of databases on multiple pages
2. **Rollups** - Calculate statistics from related databases
3. **Formulas** - Auto-calculate dates, progress percentages
4. **Templates** - Create templates for recurring pages
5. **Automations** - Set up automatic updates (Notion Plus required)

See `NOTION_STRUCTURE.md` for details on these advanced features.

---

## Support Resources

### If You Need Help

**Notion Help:**
- Notion Help Center: https://www.notion.so/help
- Database Guide: https://www.notion.so/help/guides/database-fundamentals
- Video Tutorials: https://www.youtube.com/c/Notion

**StreamGuard Resources:**
- Full setup guide: `NOTION_STRUCTURE.md`
- All page content: `NOTION_CONTENT.md`
- Project documentation: `../../` (parent directory)

---

## What You'll Have After Quick Setup

```
🛡️ StreamGuard Project
│
├── 📊 Issues Tracker (Database)
│   └── 7 issues imported
│       ├── 4 Resolved
│       ├── 2 Open
│       └── 1 In Progress
│
├── ✅ Tasks (Database)
│   └── 37 tasks imported
│       ├── 12 Completed
│       ├── 1 In Progress
│       ├── 23 Not Started
│       └── 1 Blocked
│
├── 📊 Dashboard
│   ├── Quick Stats
│   ├── Current Status
│   └── Quick Links
│
├── 📈 Project Progress
│   ├── Phase 1-5 Details
│   ├── Metrics & Statistics
│   └── Timeline
│
└── ⚡ Quick Reference Commands
    ├── Data Collection Commands
    ├── Testing Commands
    └── Troubleshooting
```

---

## Ready to Start?

Follow the 5 steps above, and you'll have a fully functional StreamGuard tracking workspace in 15 minutes!

**Pro Tip:** Bookmark the Dashboard page for quick daily access.

---

## Full Setup Option

If you want the complete setup with all features:
- Follow the detailed instructions in `NOTION_STRUCTURE.md`
- Copy all content sections from `NOTION_CONTENT.md`
- Set up all advanced database views and relations
- Estimated time: 30-45 minutes

---

**Last Updated:** October 21, 2025
**Version:** 1.0
**Status:** Ready to use

Happy tracking! 🚀
