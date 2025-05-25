---
applyTo: '**'
---
# Development Workflow and Quality Standards

## Code Quality Standards

### Version Requirements
- Python 3.13+ (as specified in pyproject.toml)
- Home Assistant Core compatibility
- Latest stable dependencies

### Linting and Formatting
- Use `ruff` for linting (matches Home Assistant core)
- Use `black` for code formatting  
- Use `mypy` for type checking
- Run `scripts/lint` before commits
- Configure pre-commit hooks for automation

### Code Organization Principles
- Single Responsibility: Each module has one clear purpose
- Type Safety: Full type annotations throughout
- Error Handling: Specific exceptions with recovery mechanisms
- Performance: Async/await patterns for I/O operations
- Maintainability: Clear naming and comprehensive documentation

## Testing Standards

### Coverage Requirements
- Minimum 90% overall test coverage
- 100% coverage for core calculation logic
- Edge case testing for all mathematical operations
- Error condition testing for all failure modes

### Test Categories
- Unit Tests: Individual function testing with mocks
- Integration Tests: Component interaction testing
- Config Flow Tests: UI configuration testing
- Entity Tests: Sensor and binary sensor testing
- Service Tests: Service call validation
- Storage Tests: Data persistence verification

### Test Data Management
- Use fixtures for reusable test data
- Mock Home Assistant services consistently
- Provide realistic historical data for prior calculations
- Test with various entity types and states
- Validate against real-world scenarios

## Documentation Standards

### Code Documentation
- Module-level docstrings for all files
- Class docstrings with purpose and usage
- Method docstrings with parameters and return values
- Inline comments for complex algorithms
- Type hints for all public interfaces

### User Documentation
- README with setup instructions
- Feature documentation in docs/ directory
- Configuration examples and best practices
- Troubleshooting guide for common issues
- Migration guides for version updates

### Technical Documentation
- Bayesian algorithm explanation
- Architecture decision records
- Performance optimization notes
- Security considerations
- API reference documentation

## Development Workflow

### Branch Management
- Use feature branches for development
- Descriptive branch names (feature/decay-handler)
- Keep branches focused and atomic
- Regular rebasing to maintain clean history

### Commit Standards
- Conventional commit messages
- Clear, concise commit descriptions
- Reference issues and pull requests
- Group related changes in single commits
- Avoid large, unfocused commits

### Code Review Process
- All changes require review
- Focus on correctness, performance, and maintainability
- Verify test coverage for new features
- Check for security implications
- Validate documentation updates

### Release Management
- Semantic versioning (MAJOR.MINOR.PATCH)
- Changelog updates for all releases
- Migration guides for breaking changes
- Tag releases with version numbers
- Archive release artifacts

## Security Considerations

### Data Handling
- No logging of sensitive information
- Secure storage of configuration data
- Validation of all user inputs
- Protection against injection attacks
- Proper error message sanitization

### Dependencies
- Regular security updates
- Minimal dependency footprint
- Trusted package sources only
- Version pinning for stability
- Security scanning of dependencies

### Home Assistant Integration
- Follow Home Assistant security guidelines
- Proper authentication handling
- Secure communication patterns
- Safe entity state handling
- Privacy-conscious data collection

## Performance Guidelines

### Calculation Efficiency
- O(n) complexity for probability calculations
- Efficient state storage and retrieval
- Minimal memory footprint
- Lazy loading of historical data
- Batched operations where possible

### Home Assistant Integration
- Async patterns for all I/O operations
- Efficient state listeners
- Minimal coordinator update frequency
- Proper entity availability handling
- Clean resource management

### Monitoring and Profiling
- Performance benchmarks for calculations
- Memory usage monitoring
- Entity update frequency tracking
- Error rate monitoring
- User experience metrics

## Maintenance Practices

### Regular Maintenance
- Dependency updates (monthly)
- Security patch reviews
- Performance optimization reviews
- Documentation updates
- Test suite maintenance

### Technical Debt Management
- Regular code quality reviews
- Refactoring for maintainability
- Legacy code removal
- Performance optimization
- Architecture improvements

### Community Engagement
- Responsive issue handling
- Feature request evaluation
- Community contribution guidance
- User support and troubleshooting
- Regular communication updates

## Deployment and Distribution

### Package Management
- Proper versioning in manifest.json
- HACS compatibility maintenance
- Release automation where possible
- Distribution via GitHub releases
- Installation documentation

### Compatibility
- Home Assistant version compatibility
- Python version support
- Dependency compatibility matrix
- Migration path documentation
- Deprecation timeline communication

### Quality Assurance
- Pre-release testing procedures
- Beta testing with community
- Automated testing in CI/CD
- Manual testing protocols
- Rollback procedures for issues
