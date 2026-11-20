"""pip-audit, in process: known vulnerabilities in the pinned dependencies. Its API is private (`pip_audit._*`),
so check this file when its version changes.

pip-audit reports no severity: its `VulnerabilityResult` holds the ID, description, fix versions, aliases and
publication date, and both of its services drop the rating the feed carries. So the rating of each advisory is
read from OSV, the feed behind pip-audit's. `Cvss3` holds the CVSS v3 standard, `Advisory` reads one of OSV's
records, `Osv` is the feed, and `PipAudit` turns what pip-audit finds into issues at the severity of their worst
advisory."""

import http.client
import json
import math
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import ClassVar, NamedTuple

from packaging.version import Version

from utils.process import run
from tools.base import Issue, Metric, Result, Step, ToolError


class Cvss3:
    """The CVSS v3 standard, as far as rating an advisory needs it: an instance is a parsed vector and the base
    score the specification's formula gives it, and the class also reads the qualitative words its rating scale
    names, for advisories that carry a word instead of a vector this scores."""

    # the base-score weights, from the specification's metric tables
    WEIGHTS: ClassVar[dict] = {
        "AV": {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2},
        "AC": {"L": 0.77, "H": 0.44},
        "UI": {"N": 0.85, "R": 0.62},
        "C": {"H": 0.56, "L": 0.22, "N": 0.0},
        "I": {"H": 0.56, "L": 0.22, "N": 0.0},
        "A": {"H": 0.56, "L": 0.22, "N": 0.0},
    }
    PRIVILEGES: ClassVar[dict] = {  # PR, whose weight also depends on whether the vulnerability changes scope
        False: {"N": 0.85, "L": 0.62, "H": 0.27},
        True: {"N": 0.85, "L": 0.68, "H": 0.50},
    }
    # Table 14, the qualitative severity rating scale, as the lowest base score in each band
    SCALE: ClassVar[dict] = {
        "CRITICAL": 9.0,
        "HIGH": 7.0,
        "MEDIUM": 4.0,
        "LOW": 0.1
    }
    ALIASES: ClassVar[dict] = {"MODERATE": "MEDIUM"}  # GitHub's word for the specification's Medium

    def __init__(self, vector: str):
        """Raises ValueError unless `vector` is a CVSS v3 vector carrying every base metric."""
        if not vector.startswith("CVSS:3"):
            raise ValueError(f"not a CVSS v3 vector: {vector}")
        metric = dict(part.split(":", 1) for part in vector.split("/")[1:])
        self.changed = metric.get("S") == "C"  # whether the flaw reaches beyond the vulnerable component
        try:
            self.weight = {m: self.WEIGHTS[m][metric[m]] for m in self.WEIGHTS}
            self.privileges = self.PRIVILEGES[self.changed][metric["PR"]]
        except KeyError as e:
            raise ValueError(f"CVSS v3 vector missing {e}: {vector}") from e

    @property
    def exploitability(self) -> float:
        """The specification's Exploitability sub-score: how reachable the flaw is."""
        return 8.22 * math.prod(self.weight[m] for m in ("AV", "AC", "UI")) * self.privileges

    @property
    def impact(self) -> float:
        """The specification's Impact sub-score: what an exploit costs. 0 when nothing is lost."""
        intact = math.prod(1 - self.weight[m] for m in ("C", "I", "A"))
        if self.changed:
            return 7.52 * (1 - intact - 0.029) - 3.25 * (1 - intact - 0.02) ** 15
        return 6.42 * (1 - intact)

    @property
    def score(self) -> float:
        """The base score: 0 when nothing is impacted, else impact and exploitability, capped at 10."""
        impact = self.impact
        if impact <= 0:
            return 0.0
        raw = min((1.08 if self.changed else 1.0) * (impact + self.exploitability), 10)
        return self.roundup(raw)

    @staticmethod
    def roundup(score: float) -> float:
        """The specification's Roundup: the smallest number with one decimal at or above `score`."""
        hundred_thousandths = round(score * 100_000)
        if hundred_thousandths % 10_000 == 0:
            return hundred_thousandths / 100_000
        return (hundred_thousandths // 10_000 + 1) / 10

    @classmethod
    def band_floor(cls, word: str) -> float | None:
        """The lowest base score in the band a qualitative rating names, for an advisory that gives a word where
        its vector is one this doesn't score. None for a word the scale doesn't name."""
        named = word.strip().upper()
        return cls.SCALE.get(cls.ALIASES.get(named, named))

    @classmethod
    def score_of(cls, vector: str) -> float | None:
        """The base score of `vector`, and None for anything this doesn't score: a CVSS v2 or v4 vector, or one
        missing a base metric. For feed data, where an unscorable vector is normal."""
        try:
            return cls(vector).score
        except ValueError:
            return None


class Advisory(NamedTuple):
    """One OSV advisory, as this reads it. `Advisory.of` is the only place that touches the feed's JSON shape,
    so the rest of the module works in IDs and scores."""

    ids: frozenset[str]  # its own ID and every ID it is an alias of: pip-audit names it by any one of them
    score: float | None  # the worst of its CVSS vectors, or the floor of the band its database's word names
    cwes: frozenset[str]  # the weaknesses its database names, where it names any: what the score cannot say
    floor: bool = False  # the score is a band's lowest, from a word: no vector this scores backs it

    @classmethod
    def merge(cls, advisories: list["Advisory"]) -> "Advisory":
        """The records that answer to one ID, read as a single advisory: the worst score and every weakness any
        of them names. Both callers group by an ID they hold, so a group is never empty."""
        scored = [a for a in advisories if a.score is not None]
        top = max((a.score for a in scored), default=None)
        return cls(
            ids=frozenset().union(*(a.ids for a in advisories)),
            score=top,
            cwes=frozenset().union(*(a.cwes for a in advisories)),
            floor=bool(scored) and all(a.floor for a in scored if a.score == top),
        )

    @classmethod
    def of(cls, record: dict) -> "Advisory":
        """The advisory an OSV record describes. A record missing the fields this reads is unrated rather than
        an error: the feed is someone else's, and a rating is worth no failed audit."""
        vectors = [Cvss3.score_of(entry.get("score", "")) for entry in record.get("severity") or []]
        words = [
            Cvss3.band_floor(str((record.get(field) or {}).get("severity", "")))
            for field
            in ("database_specific", "ecosystem_specific")
        ]
        computed = max((score for score in vectors if score is not None), default=None)
        floor = max((score for score in words if score is not None), default=None)
        return cls(
            ids=frozenset([record.get("id", ""), *(record.get("aliases") or [])]),
            score=max((score for score in (computed, floor) if score is not None), default=None),
            floor=floor is not None and (computed is None or floor > computed),
            cwes=frozenset(
                cwe
                for field
                in ("database_specific", "ecosystem_specific")
                for cwe
                in (record.get(field) or {}).get("cwe_ids") or []
            ),
        )


class Osv:
    """OSV, the advisory feed behind pip-audit's, which answers `POST /v1/query` with every advisory for a package
    version: the CVSS vectors pip-audit drops, and from some databases a severity word as well."""

    GIVE_UP = 3  # failures, with nothing answered yet, after which the feed is taken as down and not waited on

    def __init__(self, endpoint: str, timeout: int = 15):
        self.timeout = timeout
        self.endpoint = endpoint
        self.answered = False
        self.unreachable: list[str] = []  # what went wrong per dependency, for the step to report

    def records(self, name: str, version: str) -> list[dict]:
        """What OSV answers about `name==version`, raw. An unreachable OSV answers with nothing as far as this is
        concerned, which leaves the dependency unrated."""
        if not self.answered and len(self.unreachable) >= self.GIVE_UP:
            # every query so far has failed: a dependency is not worth another timeout against a feed that
            # isn't there, and a repo full of vulnerable pins would otherwise wait once per dependency
            self.unreachable.append(f"{name} {version}: not tried, OSV has not answered")
            return []
        query = {
            "package": {
                "name": name,
                "ecosystem": "PyPI"
            },
            "version": version
        }
        request = urllib.request.Request(self.endpoint, data=json.dumps(query).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                found = json.load(reply).get("vulns") or []
            self.answered = True
            return found
        except (OSError, http.client.HTTPException, ValueError) as e:
            # urlopen wraps only the request phase in URLError; reading the response raises OSError and
            # http.client errors raw, and a truncated body raises ValueError out of the JSON reader.
            # Trouble is recorded rather than raised: a rating is worth no failed audit, but a run that
            # rated nothing because OSV was down must not read as a run that found nothing to rate
            why = str(e) or e.__class__.__name__
            self.unreachable.append(f"{name} {version}: {why}")
            return []

    def advisories(self, name: str, version: str) -> list[Advisory]:
        """Every advisory OSV holds about `name==version`, read."""
        return [Advisory.of(record) for record in self.records(name, version)]

    def by_id(self, name: str, version: str) -> dict[str, Advisory]:
        """Each advisory about `name==version` under every ID it answers to, since pip-audit names an advisory
        by any one of them, with the records sharing an ID read as one."""
        records: dict[str, list] = {}
        for advisory in self.advisories(name, version):
            for ident in advisory.ids:
                records.setdefault(ident, []).append(advisory)
        return {
            ident: Advisory.merge(group)
            for ident, group
            in records.items()
        }


class PipAudit(Step):
    """Known vulnerabilities in the pinned dependencies, each rated by its advisory."""

    name = "pip-audit"
    package = "pip-audit"
    version = "2.10.1"
    policy: ClassVar[dict] = {
        "vuln-critical": "dependency with a fixable vulnerability rated critical (CVSS {critical_cvss}+)",
        "vuln-high": "dependency with a fixable vulnerability rated high (CVSS {high_cvss}+)",
        "vuln-medium": "dependency with a fixable vulnerability rated medium (CVSS {medium_cvss}+)",
        "vuln-low": "dependency with a fixable vulnerability rated low (below CVSS {medium_cvss})",
        "vuln-unrated": "fixable vulnerability no advisory rates, or that OSV could not be reached for",
        "vuln-malware": "dependency whose advisory reports malicious code in the release itself "
                        "({malware_cwes}), which outranks every rating: a version bump alone does not close it",
        "vuln-unfixed": "known vulnerability without a fixed release, whatever its rating: only dropping the "
                        "dependency closes it",
        "pins-none": "no lockfile or pinned requirements: can't audit, installs aren't reproducible",
    }
    metrics = (Metric("vulnerable_deps", "Vulnerable deps", description="pinned dependencies with known vulnerabilities"),)
    # the weaknesses that say the package itself is hostile, rather than flawed. CVSS has no metric for this,
    # so the advisory's CWE is the only machine-readable sign of it
    MALWARE: ClassVar[frozenset] = frozenset({"CWE-506", "CWE-912"})

    def __init__(self, severity: dict | None = None, settings: dict | None = None):
        """Fills in the band floors from the specification's own scale, so the policy rows and `rating` state
        them where `Cvss3` does rather than repeating the numbers here."""
        floors = {f"{band.lower()}_cvss": floor for band, floor in Cvss3.SCALE.items()}
        named = {"malware_cwes": ", ".join(sorted(self.MALWARE))}
        super().__init__(severity, {**floors, **named, **(settings or {})})

    def rating(self, score: float | None) -> str:
        """The band a CVSS base score falls in, which names the policy row: critical, high, medium or low, and
        unrated when no advisory for the dependency carries a score."""
        if score is None:
            return "unrated"
        for band in ("critical", "high", "medium"):
            if score >= self.settings[f"{band}_cvss"]:
                return band
        return "low"

    def vuln_issue(self, name: str, version: str, vulns: list, label: str, feed: Osv) -> Issue:
        """A vulnerable dependency: the worst advisory rating sets the severity, and the target is the lowest
        release that fixes all of its fixable vulnerabilities. Malicious code outranks every rating."""
        by_id = {v.id: v for v in vulns}
        ids = sorted(by_id)
        target, unfixed = fix_target(by_id, Version(version))
        matched = matched_advisories(by_id, feed.by_id(name, version))
        rated = {ident: a.score for ident, a in matched.items() if a.score is not None}
        cwes = sorted({cwe for a in matched.values() for cwe in a.cwes})
        malware = sorted(self.MALWARE.intersection(cwes))
        floors = sorted(ident for ident in rated if matched[ident].floor)
        band = self.rating(max(rated.values(), default=None))
        if malware:
            policy = "vuln-malware"
            text = malware_text(name, version, ids, malware, target)
        else:
            policy = f"vuln-{band}" if target else "vuln-unfixed"
            text = vulnerability_text(name, version, ids, rated, floors, band, target, unfixed)
        return self.issue(
            "VULN", "vulnerable-dependency", "pip-audit", policy, text.summary,
            key=f"VULN|{name.lower()}", file=label, symbol=name,
            evidence={"version": version, "vulns": ids, "fix_to": target, "rating": band, "cvss": rated,
                      "cvss_is_floor": floors, "cwe": cwes, "malware": malware,
                      "aliases": sorted({a for v in by_id.values() for a in v.aliases})[:10]},
            fix=text.fix, done_when=text.done_when)

    def measure(self, scope) -> Result:
        from pip_audit._audit import Auditor
        from pip_audit._dependency_source import DependencySourceError
        from pip_audit._service import PyPIService, ServiceError

        with tempfile.TemporaryDirectory(prefix="cleanup-audit-") as tmp:
            pinned = audit_source(scope.repo, Path(tmp))
            if pinned.source is None:
                return Result([self.issue(
                    "DEPS", "readiness", "no-pinned-dependencies", "pins-none",
                    "No lockfile or pinned requirements: dependencies can't be audited and installs aren't "
                    "reproducible", key="DEPS|pin", effort="S",
                    fix="Commit a lockfile (uv.lock) or a fully pinned requirements.txt.",
                    done_when="`vibecheck.py measure` runs pip-audit without 'not measured'.")], {},
                    skipped="no pinned source (uv.lock, pylock, pinned requirements, .venv)")
            try:
                audited = list(Auditor(PyPIService(timeout=15)).audit(pinned.source))
            except (DependencySourceError, ServiceError) as e:
                raise ToolError(str(e)) from e
        feed = Osv(endpoint="https://api.osv.dev/v1/query")
        out = [
            self.vuln_issue(dep.canonical_name, str(dep.version), vulns, pinned.label, feed)
            for dep, vulns
            in audited
            if vulns
        ]
        notes = None
        if feed.unreachable:
            notes = {"vuln_ratings": f"OSV did not answer for {len(feed.unreachable)} of {len(out)} vulnerable "
                                     f"dependencies ({feed.unreachable[0]}): those stay unrated, at high if a fix "
                                     "exists"}
        return Result(out, {"pip_audit_source": pinned.label, "vulnerable_deps": len(out)}, notes=notes)


class PinnedSource(NamedTuple):
    """Where pip-audit reads the pinned dependencies from, and what the metrics and issues call it."""

    source: object | None  # a pip-audit DependencySource; None when nothing in the repo is pinned
    label: str


def _fully_pinned(path: Path) -> bool:
    """Whether every requirement in `path` pins an exact version, so it can be audited without resolving."""
    pinned = r"^[A-Za-z0-9][\w.\-\[\], ]*==\s*[\w.+!-]+\s*(?:;.*)?(?:\\)?$"
    lines = [ln.split(" #")[0].strip() for ln in path.read_text().splitlines()]
    reqs = [ln for ln in lines if ln and not ln.startswith(("#", "--hash"))]
    return bool(reqs) and all(re.match(pinned, ln) for ln in reqs)


def audit_source(repo: Path, tmp: Path) -> PinnedSource:
    """pip-audit's dependency source for the first pinned one, and its name: uv.lock, pylock.toml, fully
    pinned requirements, .venv. No source when there is none: resolving unpinned requirements can take as
    long as installing them."""
    from pip_audit._dependency_source import PipSource, PyLockSource, RequirementSource

    if (repo / "uv.lock").exists():
        if not (uv := shutil.which("uv")):
            raise ToolError("uv.lock needs uv, which is not on PATH")
        req = tmp / "requirements.txt"
        p = run([uv, "export", "--project", repo, "--frozen", "--format", "requirements.txt",
                 "--no-emit-project", "--quiet", "-o", req])
        if p.returncode:
            raise ToolError(f"uv export failed: {p.stderr.strip()[-200:]}")
        return PinnedSource(RequirementSource([req], require_hashes=True, disable_pip=True), "uv.lock")
    if pylocks := sorted(repo.glob("pylock*.toml")):
        return PinnedSource(PyLockSource(pylocks), "pylock.toml")
    for req in [repo / "requirements.txt", *sorted(repo.glob("requirements*.txt"))]:
        if req.exists() and _fully_pinned(req):
            return PinnedSource(RequirementSource([req], no_deps=True, disable_pip=True), req.name)
    if sites := sorted(repo.glob(".venv/lib/python*/site-packages")):
        return PinnedSource(PipSource(paths=[sites[-1]]), ".venv")
    return PinnedSource(None, "")


class Wording(NamedTuple):
    """What a vulnerable-dependency issue says."""
    summary: str
    fix: str
    done_when: str


SHOWN_IDS = 3  # advisory IDs a summary lists before "..."


def some_ids(ids: list) -> str:
    """The first few advisory IDs, comma-separated, with "..." when there are more."""
    return ", ".join(ids[:SHOWN_IDS]) + (", ..." if len(ids) > SHOWN_IDS else "")


def fix_target(by_id: dict, installed: Version) -> tuple:
    """(the lowest release that fixes every fixable vulnerability or None, the IDs no release fixes). An advisory
    lists a fix per release line: only one above the installed release fixes it."""
    fixes = {
        ident: min((f for f in v.fix_versions if f > installed), default=None)
        for ident, v
        in by_id.items()
    }
    unfixed = sorted(ident for ident, f in fixes.items() if f is None)
    fixed = [f for f in fixes.values() if f is not None]
    return (str(max(fixed)) if fixed else None), unfixed


def matched_advisories(by_id: dict, held: dict) -> dict:
    """{vulnerability ID: Advisory} for the vulnerabilities OSV holds a record of, under their ID or an alias."""
    matched = {}
    for ident, vuln in by_id.items():
        known = {ident, *vuln.aliases} & held.keys()
        if known:
            matched[ident] = Advisory.merge([held[alias] for alias in known])
    return matched


def malware_text(name: str, version: str, ids: list, malware: list, target: str | None) -> Wording:
    """The wording for a release an advisory reports as malicious: bumping it is not enough."""
    start = f"Bump {name} to >= {target}" if target else f"Remove {name}: no release is named clean"
    return Wording(
        summary=(f"{name} {version}: an advisory reports malicious code in this release "
                 f"({', '.join(malware)}; {some_ids(ids)})"),
        fix=(f"{start}. A malicious release is not closed by a version bump alone: rotate every credential "
             "the build or runtime could reach, and check whether it was ever installed."),
        done_when=(f"{name} no longer resolves to an affected release, and the credentials it could reach "
                   "are rotated."),
    )


def vulnerability_text(name: str, version: str, ids: list, rated: dict, floors: list, band: str,
                       target: str | None, unfixed: list) -> Wording:
    """The wording for a flawed release: how many vulnerabilities, the worst rating and where it comes from,
    and the release that fixes them."""
    plural = "y" if len(ids) == 1 else "ies"
    summary = f"{name} {version}: {len(ids)} known vulnerabilit{plural}"
    if rated:
        worst_id = max(rated, key=rated.get)
        worst = rated[worst_id]
        basis = f"CVSS >= {worst}" if worst_id in floors else f"CVSS {worst}"
        summary += f", worst rated {band} ({basis}, {worst_id})"
    else:
        summary += f" of unrated severity ({some_ids(ids)})"
    summary += f"; fixed in {target}" if target else "; no fix yet"
    rest = f" other than {', '.join(unfixed)}, which no release fixes yet" if target and unfixed else ""
    if target:
        fix = f"Bump {name} to >= {target} in the dependency spec, then run the safety net."
    else:
        fix = "No fixed release yet: note it, or drop the dependency if it is easy."
    return Wording(summary=summary + rest, fix=fix, done_when=f"pip-audit reports no vulnerabilities for {name}{rest}.")
