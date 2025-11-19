# StreamGuard Notion Workspace Setup Guide

**Last Updated:** October 21, 2025

This guide will help you set up a comprehensive Notion workspace for tracking the StreamGuard project, including issues, tasks, progress, and documentation.

---

## Quick Setup (15 minutes)

### Step 1: Create Main Workspace

1. Open Notion and create a new page called "StreamGuard Project"
2. Add an icon (suggestion: 🛡️ or 🔒)
3. Add a cover image (optional)

### Step 2: Import Databases

1. **Issues Database:**
   - Click "+ New Database" → "Table"
   - Name it "Issues Tracker"
   - Import data from `issues.csv` (File → Import → CSV)
   - Configure views (see Database Views section below)

2. **Tasks Database:**
   - Click "+ New Database" → "Table"
   - Name it "Tasks"
   - Import data from `tasks.csv` (File → Import → CSV)
   - Configure views (see Database Views section below)

### Step 3: Create Main Pages

Create the following pages as sub-pages under "StreamGuard Project":

1. **Dashboard** (from `NOTION_CONTENT.md` → Dashboard section)
2. **Project Progress** (from `NOTION_CONTENT.md` → Project Progress section)
3. **Documentation Index** (from `NOTION_CONTENT.md` → Documentation section)
4. **Quick Reference** (from `NOTION_CONTENT.md` → Quick Reference section)
5. **Phase Details** (create 6 sub-pages for each phase)

---

## Detailed Structure

### 1. Main Workspace Layout

```
🛡️ StreamGuard Project
├── 📊 Dashboard
├── 🎯 Issues Tracker (Database)
├── ✅ Tasks (Database)
├── 📈 Project Progress
│   ├── Phase 1: CVE Collection
│   ├── Phase 2: GitHub Advisories
│   ├── Phase 3: Repository Mining
│   ├── Phase 4: Synthetic Generation
│   ├── Phase 5: Master Orchestrator
│   └── Phase 6: Model Training
├── 📚 Documentation Index
└── ⚡ Quick Reference Commands
```

---

## Database Configurations

### Issues Tracker Database

**Properties:**

| Property Name | Type | Options |
|--------------|------|---------|
| Title | Title | - |
| Status | Select | Open, In Progress, Resolved, Verified |
| Priority | Select | Critical, High, Medium, Low |
| Date Reported | Date | - |
| Date Resolved | Date | - |
| Category | Select | Unicode Encoding, Data Collection, Orchestrator, Configuration, API, Other |
| Root Cause | Text | Long text |
| Solution | Text | Long text |
| Files Modified | Multi-select | (auto-populated from CSV) |
| Related Docs | URL | - |

**Views to Create:**

1. **All Issues** (Table view)
   - Sort by: Priority (descending), Date Reported (descending)
   - Show all properties

2. **Active Issues** (Board view)
   - Group by: Status
   - Filter: Status is not "Resolved" and not "Verified"
   - Sort by: Priority (descending)

3. **Resolved Issues** (Table view)
   - Filter: Status is "Resolved" or "Verified"
   - Sort by: Date Resolved (descending)

4. **By Category** (Board view)
   - Group by: Category
   - Sort by: Priority (descending)

**Color Coding:**
- Critical: Red
- High: Orange
- Medium: Yellow
- Low: Gray

### Tasks Database

**Properties:**

| Property Name | Type | Options |
|--------------|------|---------|
| Task Name | Title | - |
| Status | Select | Not Started, In Progress, Completed, Blocked |
| Priority | Select | High, Medium, Low |
| Phase | Select | Phase 1, Phase 2, Phase 3, Phase 4, Phase 5, Phase 6, Other |
| Due Date | Date | - |
| Assigned To | Person | - |
| Dependencies | Relation | (relates to other tasks) |
| Notes | Text | Long text |
| Related Issue | Relation | (relates to Issues database) |

**Views to Create:**

1. **All Tasks** (Table view)
   - Sort by: Phase, Priority, Due Date
   - Show all properties

2. **Active Tasks** (Board view)
   - Group by: Status
   - Filter: Status is not "Completed"
   - Sort by: Priority (descending), Due Date

3. **By Phase** (Board view)
   - Group by: Phase
   - Sort by: Priority (descending)

4. **My Tasks** (Table view)
   - Filter: Assigned To is [You]
   - Filter: Status is not "Completed"
   - Sort by: Due Date, Priority

5. **Timeline** (Timeline view)
   - Group by: Phase
   - Date property: Due Date

**Color Coding:**
- High Priority: Red
- Medium Priority: Yellow
- Low Priority: Gray
- Blocked: Purple

---

## Page Templates

### Dashboard Page Structure

```markdown
# StreamGuard Project Dashboard

## Quick Stats
[Create a table with current metrics]
- Total Samples Target: 80,000
- Phases Complete: 5/6
- Issues Resolved: [X]
- Active Tasks: [X]

## Current Status
[Embed: Tasks database filtered to "In Progress"]

## Recent Issues
[Embed: Issues database, limit 5, sorted by date]

## Quick Links
- [Data Collection Guide]
- [Next Steps Guide]
- [Model Training Documentation]
```

### Project Progress Page Structure

```markdown
# Project Progress

## Overview
- **Current Phase:** 5 Complete, Phase 6 Next
- **Overall Completion:** 83% (5/6 phases)
- **Last Updated:** [Date]

## Phase Status

### ✅ Phase 1: CVE Collection (Complete)
[Toggle block with details from NOTION_CONTENT.md]

### ✅ Phase 2: GitHub Advisories (Complete)
[Toggle block with details]

[... continue for all phases ...]

## Key Metrics
[Table with statistics]

## Timeline
[Timeline of major milestones]
```

---

## Linked Databases

### How to Link Issues and Tasks

1. In Tasks database, add a "Related Issue" property (type: Relation)
2. Select "Issues Tracker" as the related database
3. Create two-way relation so issues can show related tasks

This allows you to:
- See which tasks are fixing which issues
- Track issue resolution progress through tasks
- Link documentation to both issues and tasks

---

## Regular Maintenance

### Daily Updates
- Update task statuses as work progresses
- Add new issues as they're discovered
- Update "Last Updated" dates

### Weekly Reviews
- Review "Active Issues" board
- Check task dependencies and blockers
- Update progress percentages
- Archive completed items (move to "Archived" view)

### Monthly Reviews
- Generate progress reports from databases
- Review and update documentation links
- Clean up outdated tasks
- Update phase completion status

---

## Advanced Features

### Formulas (Optional)

Add these to databases for automation:

**Issues Database - Days to Resolution:**
```
dateBetween(prop("Date Resolved"), prop("Date Reported"), "days")
```

**Tasks Database - Is Overdue:**
```
if(and(prop("Status") != "Completed", prop("Due Date") < now()), "⚠️ Overdue", "")
```

### Automations (if using Notion Plus)

1. **Auto-archive completed tasks** after 30 days
2. **Send reminders** for overdue tasks
3. **Update status** when all related tasks complete

---

## Tips for Effective Use

### Best Practices

1. **Update in Real-Time:** Change task/issue status as soon as you start/complete work
2. **Use Templates:** Create page templates for recurring documentation
3. **Link Everything:** Use @ mentions to link related pages, tasks, and issues
4. **Tag Properly:** Use consistent tagging in Categories and Phases
5. **Review Regularly:** Set aside time weekly to review and update

### Keyboard Shortcuts

- `/table` - Create new table database
- `/board` - Create board view
- `@` - Mention/link to other pages
- `//` - Add comment
- `Ctrl+/` - Show all shortcuts

---

## Troubleshooting

### CSV Import Issues

**Problem:** CSV file won't import
**Solution:** Ensure UTF-8 encoding, remove special characters

**Problem:** Dates not importing correctly
**Solution:** Use format YYYY-MM-DD in CSV

**Problem:** Select/Multi-select options not showing
**Solution:** Manually add options after import, then refresh

### Database Relations

**Problem:** Can't create two-way relation
**Solution:** Ensure both databases exist first, then create relation property

**Problem:** Related items not showing
**Solution:** Check filter settings, ensure items aren't hidden by view filters

---

## Next Steps

After setup is complete:

1. ✅ Import both CSV databases
2. ✅ Create main pages with content from NOTION_CONTENT.md
3. ✅ Configure database views as specified
4. ✅ Set up database relations (Issues ↔ Tasks)
5. ✅ Customize colors and icons
6. ✅ Invite team members (if applicable)
7. ✅ Start using for daily tracking!

---

## Support Resources

### Notion Resources
- Notion Help Center: https://www.notion.so/help
- Database Guide: https://www.notion.so/help/guides/database-fundamentals
- Templates Gallery: https://www.notion.so/templates

### StreamGuard Resources
- See `NOTION_CONTENT.md` for all page content
- See `issues.csv` and `tasks.csv` for database data
- See `quick_setup.md` for fastest setup path

---

**Estimated Setup Time:** 15-30 minutes
**Maintenance Time:** 5-10 minutes daily

**Ready to start? Follow the Quick Setup section above!**
