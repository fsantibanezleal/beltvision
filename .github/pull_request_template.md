## Summary

What this PR changes and why.

## Type

- [ ] Bug fix (no API change)
- [ ] Feature (non-breaking API addition)
- [ ] Breaking API change

## Checklist

- [ ] `ruff check beltvision tests` is clean
- [ ] `pytest -q` passes
- [ ] `beltvision synth_tear_gt --quick --out ./derived` runs end-to-end
- [ ] Heavy engines stay out of the import path (only in the `[dl]` extra, imported lazily)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] English only; no secrets committed

## Notes

Anything a reviewer should know (API impact, follow-ups).
