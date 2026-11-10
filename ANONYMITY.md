# Anonymous review package

This archive contains source, synthetic tests and configuration examples.
It contains no author/contact block, personal server path, credential, machine
UUID, Git history, environment file, execution log, trained checkpoint or
biological data payload. ZIP member paths start with `s1pert/`; timestamps and
permissions are normalized, with no archive comments or owner metadata.

Third-party project names, public URLs and source/model revisions are retained
for attribution and reproducibility. These identify external dependencies, not
the submitting authors. Statistical definitions, split seeds and scientific
code are retained.

Run `python scripts/check_release.py` after any edits and package with
`python scripts/package_release.py --out ../S1Pert_anonymous_source.zip`.
The automated scan checks paths, credentials, email addresses, IP addresses,
GPU UUIDs, syntax, file types, sizes and symlinks. It reports categories without
printing matched secret content. The release also received a separate review
for known local identity strings.

Do not publish blind-review material from a personal account. Hosting account,
repository owner, commit author and subsequent CI logs can reveal identity even
when this ZIP is clean. Use an anonymous account and review future commits and
generated outputs before publication. Never add downloaded assets, local JSON
configuration, runtime reports or local prediction files to the source tree.
