# Leaks: secrets and internal references

A credential that reaches a public repo is compromised: forks, clones and caches keep it after you
delete it. That is why the scan is mandatory and runs first.

## What `cleanup.py leaks` scans

- **The full git history:** every commit on every ref (`betterleaks git`), so a key deleted three years
  ago is still found.
- **"At HEAD"** means the matched text is somewhere in the HEAD tree, even if the file was moved after
  the commit that added it.
- **Not scanned:** untracked files (the script warns; commit or ignore them and re-run), Git LFS content,
  submodules, and anything outside git (wandb runs, checkpoints on disk, Hugging Face model cards).

## Rules

`scripts/tools/betterleaks.toml` adds a few rules to Betterleaks' ~460 built-in ones. The
built-ins already cover OpenAI, Anthropic, Hugging Face, Weights & Biases, AWS, GCP (incl.
service-account JSON), GitHub, Slack, private keys, JWTs and URLs with embedded passwords. Their
`generic-api-key` rule catches Kaggle, Comet and Neptune keys as possible secrets. The file adds:

- **`.netrc` password entries**, which `wandb login` writes.
- **Sensitive file names**, as possible credentials: `.env`, `.netrc`, `kaggle.json`, SSH keys, `*.pem`,
  `.aws/credentials`.
- **Internal references** (ids starting with `internal-`): Mila paths (`/network/scratch`, `/home/mila`),
  DRAC paths (`/project/def-…`), `#SBATCH --account`, lab hostnames and emails, wandb entities, and
  absolute home paths.

Add repo-specific patterns with `--internal REGEX` (Go RE2 syntax, no lookarounds); they are remembered.
To change the file itself:
- **The id and confidence set the severity.** An id starting with `internal-` makes a rule internal.
  Otherwise `confidence = "low"` makes it a possible credential, and no confidence a certain one.
- **Capture paths without their leading `/`.** The built-in global filter drops matches that are whole
  `/home/...` or `/Users/...` paths.
- **Use a `filter` expression** (true means drop the finding). Betterleaks 1.9 doesn't apply
  `[[rules.allowlists]]`.
- **Check it:** `betterleaks config check --config betterleaks.toml`.

## Severity

Severities come from the `TOOL` policy in `tools/betterleaks.py`, and every report.md ends with the
list. A rule's confidence separates a certain credential (critical, wherever it is) from a possible one
(high, to review); internal references matter most while they are at HEAD.

Large files are not part of this scan: `measure` checks the files at HEAD (over 5 MB is high). One
removed from HEAD stays in the history until a fresh squashed history (or a rewrite) drops it.

A credential is one issue wherever it appears, because it is rotated once. Internal references are one
issue per rule and file. Only a known public prefix (`ghp_`, `hf_`, `sk-`) and the length of a secret are ever
shown, and issue IDs come from where it first appeared, not from its value. The raw Betterleaks report
lives in a temp dir that is deleted at once.

## Triage

- Look at the location (`file`, `line`, `evidence.locations`); never print the matched value.
- Typical false positives in ML repos: dataset checksums, git SHAs in configs, wandb run IDs, base64
  test fixtures, example keys in docs.
- **False positive:** set `status` to `dismissed` with a `note`. That is all the audit needs: it
  ignores `gitleaks:allow`/`betterleaks:allow` comments and the repo's `.betterleaksignore` and
  `.gitleaksignore`, because a key waved past a pre-commit hook is exactly what it looks for. Put the
  issue's `evidence.fingerprints` in `.betterleaksignore` only for the hook you add in step 4 below.

## Remediation, in this order

1. **Rotate first.** The user revokes or rotates every confirmed credential at its provider. The agent
   lists rule, location and provider; it can't do this step. Anything ever pushed to any remote,
   including a private one, is compromised.
2. **Remove it from HEAD.** Read secrets from environment variables; add a `.env.example` and gitignore
   `.env`. Replace internal paths with config values or CLI arguments.
3. **Decide on history.** Ask the user early; it changes which findings matter.
   - **Fresh, squashed history (recommended):** publish a new repo whose first commit is the cleaned
     tree. History-only findings (except rotated credentials) then don't ship.
   - **Rewrite history:** `uvx git-filter-repo --replace-text expressions.txt` (or `--path` to drop
     large files), only with the user's explicit approval. Rewriting changes every SHA; forks and
     GitHub's caches keep the old objects. The agent never force-pushes.
4. **Prevent recurrence:** a secret-scanning pre-commit hook (`betterleaks git --pre-commit --staged`),
   a CI job running the same scan, and GitHub push protection once the repo is public.
5. **Re-scan** with `cleanup.py leaks` and the same rules.

## Betterleaks binary

`tools/betterleaks.py` pins Betterleaks 1.9.0. It downloads `betterleaks_1.9.0_<os>_<arch>.tar.gz` (darwin/linux,
x64/arm64) from the GitHub release, refuses it unless its SHA-256 matches the pinned value, and
caches it in `~/.cache/revibe-repo/` (or `$XDG_CACHE_HOME`).

- **Offline:** copy the tarball over, check its SHA-256 against the script, extract it, and set
  `BETTERLEAKS=/path/to/betterleaks`.
- **Bumping the version:** download the new release's `checksums.txt` and `checksums.txt.sigstore.json`,
  verify the signature, then update the `version` and the four `sha256` values in the `TOOL` of
  `tools/betterleaks.py`:

  ```bash
  uvx sigstore verify github --bundle checksums.txt.sigstore.json --repository betterleaks/betterleaks \
    --cert-identity "https://github.com/betterleaks/betterleaks/.github/workflows/release.yml@refs/tags/vX.Y.Z" \
    checksums.txt
  ```

- **Never pass `--validation`.** It sends candidate secrets to provider APIs to test them. Every
  confirmed credential gets rotated anyway, so knowing whether it is live changes nothing.
