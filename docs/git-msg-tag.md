# Git Commit Message Tags

Use a bracketed tag followed by a concise imperative summary:

```text
[tag]: Summary.
```

For changes larger than a small focused fix, add a blank line and list the
affected public areas with factual descriptions. Do not claim measurements,
performance, energy, or PPA evidence that the repository does not contain.

- `[artifact]`: A reviewed public mechanism-artifact delivery spanning source,
  tests, and scope documentation.
- `[core]`: Functional PPIM, HCF, dataflow, runtime, or rendering behavior.
- `[rtl]`: Chisel RTL or RTL verification behavior.
- `[docs]`: Documentation-only changes.
- `[test]`: Test-only changes.
- `[build]`: Packaging, dependency, or verification-entry-point changes.
- `[chore]`: Repository maintenance without public mechanism changes.

Use the narrowest tag that accurately describes the commit. A mixed public
artifact delivery uses `[artifact]` and names its primary areas in the body.
