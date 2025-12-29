# Generate Release Notes

Generate release notes by analyzing the formatting style from recent CHANGELOG.md entries and processing commit_messages.txt and release_diff.txt.

## Objective

- Extract formatting patterns from the last 2-3 **stable release** entries in CHANGELOG.md (skip pre-releases)
- Parse commit_messages.txt to extract commits, categorize them, and identify PR links
- Analyze release_diff.txt for breaking changes and major features
- Generate formatted release notes matching the CHANGELOG.md style exactly

## Requirements

- `scripts/notes` must have been executed to generate `commit_messages.txt` and `release_diff.txt`
- `CHANGELOG.md` must exist with recent release entries to analyze style
- User should provide version number and release date (or detect from git if not provided)
- Repository should be accessible to determine previous release version for Full Changelog link

## Steps

### Step 1: Analyze CHANGELOG.md Formatting Style

1. Read the last 2-3 **stable release** entries from `CHANGELOG.md` (starting from `## [Version]` headers)
   - **Important**: Skip pre-releases (entries with `-pre` in the version number, e.g., `2025.12.4-pre2`)
   - Only analyze stable releases (e.g., `2025.12.3`, `2025.12.2`, `2025.12.1`)
   - Pre-releases may have different formatting (warning blocks, pre-release notices) that should not be used as style reference for stable releases
2. Identify and document the formatting patterns:

   - **Header format**: `## [VERSION] - YYYY-MM-DD` (note exact spacing and brackets)
   - **Summary/intro paragraphs**: Optional introductory text after header, before sections
   - **Warning blocks**: Format like `> [!CAUTION]`, `> [!NOTE]`, `> [!WARNING]` with proper indentation
   - **"What's Changed" section**: Header `## What's Changed` followed by bullet points
   - **Bullet point format**: `* [Description] by @username in https://github.com/.../pull/XXX`
     - Note: Description should be concise, start with action verb (Add, Fix, Update, etc.)
     - Username format: `@username` (no space before @)
     - PR link format: `https://github.com/Hankanman/Area-Occupancy-Detection/pull/XXX`
   - **"New Contributors" section**: Header `## New Contributors` followed by bullet points
     - Format: `* @username made their first contribution in https://github.com/.../pull/XXX`
   - **Full Changelog link**: Format `**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/PREVIOUS_VERSION...NEW_VERSION`
   - **Spacing**: Note blank lines between sections (typically one blank line)

3. Document any special formatting:
   - How version-only commits are handled (often filtered out or placed at end)
   - How dependency bumps are formatted (often grouped together)
   - How merge commits are formatted
   - How breaking changes are indicated
   - **Note**: Pre-releases may include warning blocks (`> [!CAUTION]`, `> [!WARNING]`) that should NOT be included in stable release notes unless there are actual breaking changes

### Step 2: Parse commit_messages.txt

1. Read `commit_messages.txt` and parse the format:

   - Commits are separated by `---` on its own line
   - Each commit starts with a commit hash (7 characters)
   - Followed by commit title (single line)
   - Followed by optional description (bullet points starting with `-`)
   - Empty lines separate sections

2. For each commit, extract:

   - **Commit hash**: First line (7 characters)
   - **Title**: Second line (main commit message)
   - **Description**: Lines starting with `-` (bullet points)
   - **PR number**: Look for patterns like:
     - `#XXX` in commit message
     - `pull/XXX` in commit message
     - Merge commits: `Merge pull request #XXX`
   - **Author**: Look for patterns like:
     - `by @username` in commit message
     - Merge commits: `from username/branch`
     - Default to `@Hankanman` if not found

3. Categorize each commit based on title and description keywords:

   - **Features**: Keywords: "add", "implement", "introduce", "new", "support", "create", "enable"
   - **Bug Fixes**: Keywords: "fix", "bug", "error", "issue", "resolve", "correct", "repair"
   - **Refactors**: Keywords: "refactor", "improve", "enhance", "update", "simplify", "optimize", "extract", "restructure"
   - **Dependencies**: Keywords: "bump", "update dependencies", "upgrade"
   - **Version Updates**: Keywords: "version", "2025.", "2024." (these should be filtered out from main content or placed at end)
   - **Documentation**: Keywords: "docs", "documentation", "readme", "update descriptions"
   - **Tests**: Keywords: "test", "coverage", "unit test"

4. Identify new contributors:

   - Look for merge commits with different authors than `@Hankanman`
   - Look for commits with "by @username" where username is not `Hankanman`
   - Track which PRs are first contributions

5. Filter out version-only commits:
   - Commits that only update version numbers in files
   - These should be excluded from "What's Changed" or placed at the end

### Step 3: Analyze release_diff.txt (REQUIRED - Critical for Breaking Changes)

**This step is MANDATORY** - The diff file contains critical information about breaking changes that must be documented.

1. **Read `release_diff.txt` thoroughly** to understand the scope of changes:

   - Use grep or search to find specific patterns
   - Pay special attention to migration files, database schema files, and configuration files
   - Look for version number changes (CONF_VERSION, DB_VERSION)

2. **Search for breaking changes indicators** (use grep with case-insensitive search):

   a. **Database and Migration Changes**:

   - Search for: `migration`, `migrate`, `CONF_VERSION`, `DB_VERSION`, `db_version`
   - Look for: `delete.*database`, `recreate.*database`, `reset.*database`, `schema.*version`
   - Check migration files (`migrations.py`) for database reset logic
   - Look for comments like "delete database", "recreate database", "reset database"
   - Check for `async_reset_database_if_needed` or similar functions
   - Look for version increments in comments (e.g., "CONF_VERSION: Final = 15" → "CONF_VERSION: Final = 16")

   b. **Timestamp and Timezone Changes**:

   - Search for: `timestamp`, `timezone`, `datetime`, `UTC`, `local.*time`
   - Look for: New time utility files (`time_utils.py`, `time.py`)
   - Check for changes to datetime handling functions
   - Look for comments about timezone normalization

   c. **Schema and Data Structure Changes**:

   - Search for: `schema`, `Column`, `Table`, `Base`, `__tablename__`
   - Look for: New database tables, modified column types, removed columns
   - Check for changes to data models and ORM classes

   d. **Configuration Format Changes**:

   - Search for: `CONF_`, `config`, `options`, `data`
   - Look for: Removed configuration keys, renamed keys, new required fields
   - Check config_flow.py for schema changes

   e. **API and Function Signature Changes**:

   - Look for: Function definitions with changed parameters
   - Search for: `def` followed by parameter changes
   - Check for removed functions or classes

3. **Detect major features**:

   - New files/modules added (look for `+++ b/path/to/new_file.py`)
   - Significant new functionality (large additions, new classes, new modules)
   - New utility functions or helper modules
   - New service endpoints or API additions

4. **Note performance improvements**:

   - Caching additions (look for "cache", "caching")
   - Optimization keywords ("optimize", "performance", "improve")
   - Database query improvements (look in `db/queries.py` or similar)
   - Validation and deduplication logic

5. **Extract specific technical details**:

   - Note what changed and why (from comments and commit messages)
   - Identify the impact on users (data loss, configuration changes, etc.)
   - Document any automatic migration or rebuilding processes
   - Note any warnings or cautions mentioned in code comments

6. **Document findings**:
   - Create a list of breaking changes with explanations
   - Note which changes require user action vs. automatic handling
   - Identify data loss scenarios (database resets, cleared caches, etc.)
   - Note migration paths and automatic recovery mechanisms

### Step 4: Determine Version and Date

1. Ask user for version number if not provided, or detect from:
   - Git tags (latest tag)
   - `pyproject.toml` version field
   - `custom_components/area_occupancy/manifest.json` version field
2. Ask user for release date if not provided, or use today's date in YYYY-MM-DD format
3. Determine previous version:
   - From git tags (second-to-last **stable** release, skip pre-releases)
   - Or from CHANGELOG.md (last **stable** release entry, skip pre-releases)
   - This is needed for the Full Changelog link

### Step 5: Generate Release Notes

**Important for Stable Releases**: Stable releases should be more detailed and user-focused than pre-releases. Include comprehensive explanations, user benefits, and context about what changed and why it matters.

1. **Create header**: `## [VERSION] - YYYY-MM-DD` (match exact format from CHANGELOG.md)

2. **Add summary/intro paragraph** (ALWAYS include for stable releases):

   - **For stable releases**: Write a detailed, user-focused summary paragraph (2-5 sentences)
   - Explain what the release adds, improves, or fixes in user-friendly language
   - Focus on what users will notice and benefit from
   - Include specific examples of new features or improvements
   - Use bullet points if listing multiple key changes (as seen in stable releases like 2025.12.3)
   - Format: Start with "This version adds:" or "Minor release to..." or similar, followed by bullet points or sentences
   - Use similar tone and style as recent stable release entries in CHANGELOG.md
   - Make it informative and helpful for users deciding whether to upgrade

3. **Add "Potential Breaking Changes" section** (REQUIRED if breaking changes detected):

   - **MANDATORY**: If Step 3 analysis detected ANY breaking changes, you MUST include a `## ‼️Potential Breaking Changes` section
   - **Format**: Follow the exact format from stable releases in CHANGELOG.md (see 2025.12.3 example)
   - **Structure**:
     - Start with: `## ‼️Potential Breaking Changes`
     - Add introductory text: `The following applies _**if**_ you are upgrading from a version _**before**_ [NEW_VERSION].`
     - Add warning blocks using the format from stable releases:
       - `> [!CAUTION]` for critical breaking changes (database resets, data loss)
       - `> [!NOTE]` for important information (automatic migrations, rebuilding processes)
       - `> [!WARNING]` for warnings (automation impacts, configuration changes)
   - **Content requirements**:
     - **Database resets**: If database is reset (CONF_VERSION change, schema changes), clearly explain:
       - That the database file will be deleted and recreated
       - What data will be lost (intervals, priors, correlations, etc.)
       - That data will be automatically rebuilt from Recorder history
       - How long rebuilding might take
     - **Migration logic changes**: Explain what changed and why fresh installations benefit
     - **Configuration changes**: Note any removed or renamed configuration options
     - **API changes**: Note any function signature changes or removed APIs
   - **If NO breaking changes detected**: Skip this section entirely (do not include empty section)
   - **Important**: Only reference warning block formatting from stable releases, not pre-releases

4. **Create feature sections** (for stable releases with significant features):

   - **For stable releases with major features**: Consider adding a "🎉 New & Updated Features" section before "What's Changed"
   - List key user-facing features with bullet points
   - Focus on what users can do or what has improved
   - Use user-friendly language, not technical jargon
   - Only include if there are substantial new features (like 2025.12.1 example)

5. **Create "What's Changed" section**:

   - Header: `## What's Changed` (or `## 📄 What's Changed` if feature section was added)
   - Group commits by category (Features first, then Bug Fixes, then Refactors, then Dependencies)
   - Format each bullet point: `* [Description] by @username in https://github.com/Hankanman/Area-Occupancy-Detection/pull/XXX`
   - **Description formatting rules for stable releases**:
     - Start with action verb (Add, Fix, Update, Implement, etc.)
     - Capitalize first letter
     - **For stable releases**: Provide more context and explanation (can be longer than pre-releases)
     - Use present tense ("Add feature" not "Added feature")
     - **For stable releases**: Include user-facing benefits when possible
     - If commit has detailed description, expand on the main point to explain what it does for users
     - Make descriptions informative - explain not just what changed, but why it matters
   - **PR link formatting**:
     - Extract PR number from commit message
     - Format as: `https://github.com/Hankanman/Area-Occupancy-Detection/pull/XXX`
     - If no PR number found, omit the link (just description and author)
   - **Author formatting**:
     - Use `@username` format
     - Default to `@Hankanman` if not found
     - For dependabot: Use `@dependabot[bot]`
   - **Ordering**:
     - Features first
     - Bug fixes second
     - Refactors third
     - Dependencies last
     - Within each category, order by importance (major changes first)

6. **Create "New Contributors" section** (if any):

   - Header: `## New Contributors`
   - Format: `* @username made their first contribution in https://github.com/Hankanman/Area-Occupancy-Detection/pull/XXX`
   - Only include contributors who haven't contributed before (check against recent CHANGELOG entries)

7. **Add Full Changelog link**:

   - Format: `**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/PREVIOUS_VERSION...NEW_VERSION`
   - Replace PREVIOUS_VERSION with the last release version
   - Replace NEW_VERSION with the new version

8. **Add spacing**:
   - One blank line between sections
   - One blank line after Full Changelog link

### Step 6: Review and Refine

1. **Review the generated release notes**:

   - **Breaking changes verification**:
     - Verify that ALL breaking changes found in Step 3 are documented in the "Potential Breaking Changes" section
     - Ensure database resets are clearly explained with impact on users
     - Check that migration logic changes are properly documented
     - Verify that data loss scenarios are clearly explained
   - **Formatting**:
     - Check formatting matches CHANGELOG.md style exactly
     - Verify warning blocks use correct format (`> [!CAUTION]`, `> [!NOTE]`, `> [!WARNING]`)
     - Ensure proper spacing between sections
   - **Content accuracy**:
     - Verify all PR links are correct
     - Ensure descriptions are clear and concise
     - Check that version-only commits are filtered appropriately
     - Verify contributor information is accurate
     - Ensure technical details from diff are accurately reflected

2. **Make any necessary adjustments**:
   - Fix formatting inconsistencies
   - Improve descriptions for clarity
   - Ensure proper categorization
   - Add missing breaking changes if any were overlooked
   - Enhance breaking change explanations with more detail if needed
   - Add or remove sections as needed

## Output

Generate formatted release notes ready to be:

- Copied into GitHub release notes
- Added to CHANGELOG.md (inserted after `## [Unreleased]` section)
- Used for release announcements

The output should match the exact formatting style of recent **stable release** entries in CHANGELOG.md, including:

- Proper markdown formatting
- Correct section headers
- Properly formatted bullet points with PR links
- Appropriate spacing
- Warning blocks if needed
- Full Changelog link

## Example Output Format

```markdown
## [2025.12.4] - 2025-12-29

This version stabilizes the timezone normalization changes introduced in the pre-releases and includes several important improvements:

- **Improved timezone handling** ensures accurate occupancy detection across different time zones, fixing issues with timestamp normalization that could affect data accuracy. All datetime operations now consistently use UTC internally with proper conversion to local time for display and analysis.
- **Enhanced migration logic** prevents unnecessary migrations for fresh installations, making setup smoother for new users. The migration system now intelligently detects fresh entries and skips migration steps that aren't needed.
- **Added support for additional sensor types** including VOLATILE_ORGANIC_COMPOUNDS_PARTS, expanding the range of environmental sensors that can be used for occupancy detection.
- **Improved interval caching** with validation and deduplication for better performance and reliability. Invalid intervals (where start time is greater than end time) are now automatically skipped, and duplicate intervals are prevented from being stored.
- **Updated user interface text** and descriptions for clearer configuration options, making it easier to understand door and wasp-in-box sensor settings.
- **Refactored binary sensor state mapping** to improve consistency and maintainability across the integration. The mapping logic has been extracted to a centralized utility function.

## ‼️Potential Breaking Changes

The following applies _**if**_ you are upgrading from a version _**before**_ 2025.12.4.

> [!CAUTION] > **Database Reset Required**: This version includes fundamental changes to how timestamps are handled throughout the integration. The database schema version has been incremented (CONF_VERSION 15 → 16) to support timezone normalization and local bucketing. **Your `area_occupancy.db` file will be automatically deleted and recreated with the new schema.** All historical data (intervals, priors, correlations) will be cleared and will need to be rebuilt from Home Assistant's Recorder history.

> [!NOTE]
> The database reset is necessary because the new timezone normalization system requires consistent UTC storage with proper timezone-aware datetime handling. Existing data stored with the old timestamp format cannot be safely migrated. The integration will automatically rebuild historical data from your Home Assistant Recorder over the next few hours/days.

> [!WARNING]
> If you rely on historical occupancy data for automations or analysis, be aware that this data will be lost during the upgrade. The integration will rebuild this data automatically, but it may take some time depending on how much history is available in your Recorder.

## What's Changed

- Implement timezone normalization and local bucketing utilities by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/304
- Add map_binary_state_to_semantic utility function and update entity logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/317
- Update device class in environmental section schema to include additional sensor type by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/318
- Add validation and deduplication for occupied interval caching by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/316
- Enhance migration logic and version handling for Area Occupancy integration by @Hankanman
- Update descriptions for door and wasp-in-box states in strings and translations files for clarity and consistency by @Hankanman
- Improving user interface text by @simon5738 in https://github.com/Hankanman/Area-Occupancy-Detection/pull/307

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.12.3...2025.12.4
```

## Notes

- **CRITICAL: Diff Analysis is Mandatory**: Step 3 (analyzing release_diff.txt) is REQUIRED, not optional. Breaking changes MUST be detected and documented.
- **Breaking Changes Detection**: Always search the diff for:
  - Database version changes (CONF_VERSION increments)
  - Migration logic changes (especially database reset logic)
  - Timestamp/timezone handling changes
  - Schema modifications
  - Configuration format changes
- **For stable releases**: Always provide detailed, user-focused explanations
  - Include comprehensive summary paragraphs explaining what changed and why it matters
  - Use bullet points in the summary to highlight key improvements
  - Explain user benefits, not just technical changes
  - Make descriptions informative and helpful for users deciding whether to upgrade
- **Breaking Changes Documentation**: When breaking changes are detected:
  - MUST include "Potential Breaking Changes" section
  - Clearly explain database resets, data loss, and automatic recovery
  - Use appropriate warning blocks (`> [!CAUTION]`, `> [!NOTE]`, `> [!WARNING]`)
  - Explain impact on users and what actions (if any) they need to take
- Always match the exact formatting style from **stable releases** in CHANGELOG.md (skip pre-releases for style reference)
- Filter out version-only commits from main content (or place at end)
- Group dependency bumps together
- Use consistent capitalization and punctuation
- Ensure PR links are valid GitHub URLs
- Verify contributor usernames are correct
- Include Full Changelog link with correct version comparison
- Do NOT include pre-release warning blocks in stable release notes unless there are actual breaking changes
- **Stable releases should be more detailed than pre-releases** - focus on user value and clear explanations
