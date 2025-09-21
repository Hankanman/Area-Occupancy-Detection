# Database Locking Solution for Area Occupancy Detection

## Problem Statement

The Area Occupancy Detection integration was experiencing database corruption due to multiple instances of the integration running concurrently against the same SQLite database file. The existing file lock implementation was not being used in critical database operations, leading to race conditions and data corruption.

## Root Cause Analysis

1. **No file locking in critical operations**: Database operations like `save_data()`, `load_data()`, and `sync_states()` were accessing the database without any locking mechanism.

2. **Multiple coordinator instances**: Each Home Assistant restart or reload creates new coordinator instances that can access the same database file simultaneously.

3. **Race conditions**: Multiple instances could write to the database at the same time, causing corruption.

## Solution Implementation

### 1. Enhanced FileLock Class

**File**: `custom_components/area_occupancy/utils.py`

Enhanced the `FileLock` class with:

- **Stale lock detection**: Automatically removes lock files older than 5 minutes
- **Better logging**: Added debug logging for lock acquisition/release
- **Improved error handling**: Better timeout and exception handling
- **Atomic operations**: Uses `O_EXCL` flag for atomic file creation

```python
class FileLock:
    """Robust file-based lock using context manager with atomic file creation and stale lock detection."""

    def __init__(self, lock_path: Path, timeout: int = 60, stale_lock_timeout: int = 300):
        # ... implementation
```

### 2. Database Locking Context Manager

**File**: `custom_components/area_occupancy/db.py`

Added a new `get_locked_session()` method that wraps all database operations with file locking:

```python
@contextmanager
def get_locked_session(self, timeout: int = 30):
    """Get a database session with file locking to prevent concurrent access."""
    # ... implementation
```

### 3. Updated Critical Database Operations

All critical database operations now use the locked session:

- **`load_data()`**: Uses `get_locked_session()` instead of `get_session()`
- **`save_area_data()`**: Uses `get_locked_session()` with session.merge() for upsert operations
- **`save_entity_data()`**: Uses `get_locked_session()` with session.merge() for upsert operations
- **`sync_states()`**: Uses `get_locked_session()` for interval synchronization
- **`get_area_data()`**: Uses `get_locked_session()` for data retrieval
- **`get_latest_interval()`**: Uses `get_locked_session()` for query operations

### 4. Improved Error Handling

- **Timeout handling**: Proper `HomeAssistantError` exceptions when database is busy
- **Fallback behavior**: Falls back to regular session if lock path is not available
- **Graceful degradation**: System continues to work even if locking fails

## Key Features

### 1. Atomic File Locking

- Uses `O_CREAT | O_EXCL | O_WRONLY` flags for atomic file creation
- Only one process can create the lock file at a time
- Prevents race conditions in lock acquisition

### 2. Stale Lock Detection

- Automatically detects and removes stale lock files
- Configurable stale lock timeout (default: 5 minutes)
- Prevents deadlocks from crashed processes

### 3. Comprehensive Coverage

- All critical database operations are protected
- Both read and write operations use locking
- Migration operations already had locking (maintained)

### 4. Performance Optimized

- Minimal overhead with short lock acquisition timeouts
- Uses session.merge() for efficient upsert operations
- Proper connection pooling maintained

## Testing

Created comprehensive tests in `tests/test_database_locking.py`:

1. **FileLock Basic Functionality**: Tests lock creation, acquisition, and release
2. **Concurrent Access Prevention**: Verifies that only one thread can acquire a lock at a time
3. **Timeout Handling**: Tests proper timeout behavior when lock is held
4. **Stale Lock Detection**: Tests automatic removal of stale locks
5. **Database Integration**: Tests that database operations use locking

## Usage

### For Developers

The locking is automatically applied to all critical database operations. No additional code changes are needed for new database operations - simply use `get_locked_session()` instead of `get_session()`.

```python
# Old way (not locked)
with self.get_session() as session:
    # database operations

# New way (locked)
with self.get_locked_session() as session:
    # database operations
```

### For Users

The locking is completely transparent to users. The integration will:

- Automatically prevent database corruption
- Handle multiple instances gracefully
- Provide better error messages if database is busy
- Continue working even if locking fails

## Configuration

The locking behavior can be configured through the FileLock constructor:

- **`timeout`**: Maximum time to wait for lock acquisition (default: 30 seconds)
- **`stale_lock_timeout`**: Time after which a lock is considered stale (default: 5 minutes)

## Benefits

1. **Prevents Database Corruption**: Eliminates race conditions that cause corruption
2. **Handles Multiple Instances**: Allows multiple coordinator instances to coexist safely
3. **Automatic Recovery**: Handles stale locks from crashed processes
4. **Minimal Performance Impact**: Locking overhead is negligible
5. **Backward Compatible**: Existing functionality remains unchanged
6. **Robust Error Handling**: Graceful degradation when locking fails

## Monitoring

The solution includes comprehensive logging:

- Lock acquisition and release events
- Timeout warnings
- Stale lock removal notifications
- Error conditions

Monitor logs for:

- `Acquired database lock`
- `Released database lock`
- `Removing stale lock file`
- `Database is busy, please try again later`

## Future Enhancements

1. **Distributed Locking**: Consider Redis-based locking for multi-instance deployments
2. **Lock Metrics**: Add metrics for lock acquisition times and contention
3. **Adaptive Timeouts**: Dynamic timeout adjustment based on system load
4. **Lock Priority**: Priority-based locking for critical operations

## Conclusion

This database locking solution provides a robust, efficient, and transparent way to prevent database corruption in the Area Occupancy Detection integration. It addresses the root cause of corruption while maintaining excellent performance and user experience.
