"""Rule-based Werewolf agent for make.fun's werewolf-agent-01 bounty.

Plays one seat in 30 games of 8-player Werewolf through the env at
MAKEFUN_ENV_URL (POST /observe, /act). No model calls: every decision is
a pure function of the game view the env serves, so a replay of the same
views reproduces the same actions byte for byte. Randomness, where a
tie must be broken or a phrasing chosen, comes from a hash of
MAKEFUN_TRIAL_SEED, the game number and the view itself.

Strategy in one paragraph: parse every public statement for role claims
(seer claims and their nightly check reports above all), score each
claimant's trust by how their reports line up with the roles revealed
by deaths and exiles, keep a suspicion score for every seat from
trusted seer reports, voting records and defence patterns, and let
each role act on that table: the seer reveals a found werewolf at once
with its full history, the doctor guards the trusted seer, villagers
push the best-supported target and coordinate on one exile, and the
werewolf matches its partner's night pick, kills confirmed villagers
and the seer, steers the day vote onto villagers and fake-claims seer
only when cornered or when one framed exile ends the game.
"""

import hashlib
import json
import os
import re
import urllib.request

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
TRIAL_SEED = os.environ.get("MAKEFUN_TRIAL_SEED", "0")

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(path, body):
    req = urllib.request.Request(
        ENV_URL + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with _opener.open(req, timeout=120) as res:
        return json.loads(res.read().decode("utf-8"))


def rng(*parts):
    """Deterministic floats in [0, 1) keyed by the given parts."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return [int.from_bytes(h[i:i + 8], "big") / float(1 << 64) for i in range(0, 32, 8)]


# ---------------------------------------------------------------- parsing

SEER_CLAIM = re.compile(
    r"\bi(?:'m| am) the seer\b|\bi am seer\b|\bseer here\b"
    r"|\bas (?:the|your) seer\b|\bi am a seer\b",
    re.I,
)
DOCTOR_CLAIM = re.compile(r"\bi(?:'m| am) the doctor\b|\bi am doctor\b", re.I)
WOLF_WORDS = re.compile(r"werewol(f|ves)\b|\bwolf\b", re.I)
CLEAN_WORDS = re.compile(
    r"not a (?:werewolf|wolf)\b|\bclean\b|\binnocent\b|\bvillager\b"
    r"|\bcleared\b|\bhuman\b|\btown\b",
    re.I,
)
CHECK_VERBS = re.compile(r"check|inspect|scry|scanned|looked at|saw", re.I)
SEAT_REF = re.compile(r"seat\s*(\d+)", re.I)
WILL_VOTE = re.compile(
    r"(?:will|would|going to|shall)\s+vote|vote\s+(?:for|to exile)", re.I)


def seat_ref(name, seat):
    return "seat %d (%s)" % (seat, name)


class Game:
    """Everything one decision needs, rebuilt from the view alone."""

    def __init__(self, view):
        self.view = view
        self.day = view["day"]
        self.phase = view["phase"]
        self.me = view["you"]["seat"]
        self.my_role = view["you"]["role"]
        self.my_name = view["you"]["name"]
        self.partner = None
        if view["you"].get("partner"):
            self.partner = view["you"]["partner"]["seat"]
        self.name = {}
        self.alive = {}
        self.revealed = {}  # seat -> role, only once the env reveals it
        for p in view["players"]:
            self.name[p["seat"]] = p["name"]
            self.alive[p["seat"]] = p["alive"]
            if p.get("role"):
                self.revealed[p["seat"]] = p["role"]
        self.log = view["log"]
        self.speeches = []  # (day, round, seat, text)
        self.votes = []     # (day, voter, target or None), in cast order
        self.exiles = []    # (day, seat, role) of exiled players
        self.deaths = []    # (day, seat, role) of night deaths
        for e in self.log:
            k = e["kind"]
            if k == "speech":
                self.speeches.append((e["day"], e["round"], e["seat"], e["text"]))
            elif k == "vote":
                self.votes.append((e["day"], e["seat"], e["target"]))
            elif k == "exile" and e.get("seat") is not None:
                self.exiles.append((e["day"], e["seat"], e.get("role")))
            elif k == "dawn" and e.get("died") is not None:
                self.deaths.append((e["day"], e["died"], e.get("role")))
        self.parse_claims()
        self.rate_claims()
        self.suspicion()

    def targets_in(self, text):
        """Seats a text points at, by 'seat N' or by player name."""
        found = []
        for m in SEAT_REF.finditer(text):
            s = int(m.group(1))
            if s in self.name and s not in found:
                found.append(s)
        low = " " + text.lower() + " "
        for s, nm in self.name.items():
            n = nm.lower()
            if s in found:
                continue
            if (" " + n + " ") in low or ("(" + n + ")") in low or \
               (" " + n + ",") in low or (" " + n + ".") in low:
                found.append(s)
        return found

    def parse_claims(self):
        self.seer_claimants = {}   # seat -> {"day": int, "checks": {seat: (is_wolf, day)}}
        self.doctor_claimants = set()
        self.accusations = []      # (day, accuser, target) suspicion, not a check
        for (day, rnd, seat, text) in self.speeches:
            if SEER_CLAIM.search(text):
                self.seer_claimants.setdefault(seat, {"day": day, "checks": {}})
            if DOCTOR_CLAIM.search(text):
                self.doctor_claimants.add(seat)
            for sent in re.split(r"[.!?;]", text):
                ts = self.targets_in(sent)
                if not ts:
                    continue
                is_check = seat in self.seer_claimants and (
                    CHECK_VERBS.search(sent) or "night" in sent.lower())
                for t in ts:
                    if t == seat:
                        continue
                    wolf_hit = WOLF_WORDS.search(sent)
                    clean_hit = CLEAN_WORDS.search(sent)
                    if is_check and clean_hit:
                        self.seer_claimants[seat]["checks"][t] = (False, day)
                    elif is_check and wolf_hit:
                        self.seer_claimants[seat]["checks"][t] = (True, day)
                    elif wolf_hit and not clean_hit and \
                            re.search(r"suspicious|suspect|accuse|lying|fake|"
                                      r"must be|seems|acting", sent, re.I):
                        self.accusations.append((day, seat, t))

    # -- trust -----------------------------------------------------------

    def rate_claims(self):
        """Score each seer claimant: confirmed, plausible or exposed."""
        self.trust = {}  # claimant seat -> level
        for seat, c in self.seer_claimants.items():
            level = "plausible"
            if self.revealed.get(seat) is not None:
                level = "confirmed" if self.revealed[seat] == "seer" else "exposed"
            for (t, (is_wolf, _d)) in c["checks"].items():
                r = self.revealed.get(t)
                if r is None:
                    continue
                if is_wolf and r == "werewolf":
                    level = "confirmed"
                elif is_wolf and r != "werewolf":
                    level = "exposed"
                elif not is_wolf and r == "werewolf":
                    level = "exposed"
                elif not is_wolf and r != "werewolf" and level == "plausible":
                    level = "backed"
            self.trust[seat] = level
        # Two claimants reporting opposite verdicts on one seat: the one
        # whose other reports already failed is the liar; when neither has
        # failed, both stay in a claim war.
        seats = list(self.seer_claimants)
        for i in range(len(seats)):
            for j in range(i + 1, len(seats)):
                a, b = seats[i], seats[j]
                ca = self.seer_claimants[a]["checks"]
                cb = self.seer_claimants[b]["checks"]
                for t in set(ca) & set(cb):
                    if ca[t][0] != cb[t][0]:
                        if self.trust[a] == "exposed" or self.trust[b] == "exposed":
                            continue
                        self.trust[a] = "war"
                        self.trust[b] = "war"

    def trusted_seer(self):
        """The claimant the village can rely on, if any."""
        for level in ("confirmed", "backed"):
            for seat, lv in self.trust.items():
                if lv == level and seat != self.me and self.alive.get(seat):
                    return seat
        return None

    def sole_claimant(self):
        alive_claims = [s for s in self.seer_claimants
                        if self.alive.get(s) and self.trust.get(s) != "exposed"]
        return alive_claims[0] if len(alive_claims) == 1 else None

    def my_checks(self):
        """As the real seer, my own results as {seat: is_wolf}."""
        out = {}
        for r in (self.view.get("seerResults") or []):
            out[r["seat"]] = bool(r["werewolf"])
        return out

    # -- suspicion ---------------------------------------------------------

    def suspicion(self):
        """Village-side suspicion per seat, from public information."""
        s = {seat: 0.0 for seat in self.name}
        claimants = [c for c in self.seer_claimants if self.trust.get(c) != "exposed"]
        sole = len([c for c in claimants if self.alive.get(c)]) == 1
        for c in claimants:
            lv = self.trust[c]
            for (t, (is_wolf, _d)) in self.seer_claimants[c]["checks"].items():
                if not self.alive.get(t):
                    continue
                if is_wolf:
                    s[t] += {"confirmed": 1000, "backed": 220, "plausible": 120,
                             "war": 45}.get(lv, 30) * (1.0 if sole or lv in
                             ("confirmed", "backed") else 0.6)
                else:
                    s[t] -= {"confirmed": 1000, "backed": 200, "plausible": 110,
                             "war": 40}.get(lv, 30)
            if lv == "war":
                s[c] += 30  # one of the two claimants is a wolf
        for c, lv in self.trust.items():
            if lv == "exposed" and self.alive.get(c):
                s[c] += 500  # a proven fake seer is a werewolf
        # Voting record against revealed roles.
        for (day, voter, target) in self.votes:
            if target is None or not self.alive.get(voter):
                continue
            r = self.revealed.get(target)
            if r is None:
                continue
            if r == "werewolf":
                s[voter] -= 14
            else:
                s[voter] += 11
        # Speech pressure: accusing players who flipped village, or
        # clearing players who flipped wolf, both smell.
        for (day, acc, t) in self.accusations:
            r = self.revealed.get(t)
            if r is not None and r != "werewolf" and self.alive.get(acc):
                s[acc] += 7
        for c in self.seer_claimants:
            for (t, (is_wolf, _d)) in self.seer_claimants[c]["checks"].items():
                pass  # handled above; claims are not accusations
        self.sus = s

    def ref(self, seat):
        return seat_ref(self.name[seat], seat)

    def living_others(self):
        return [s for s in self.name if self.alive.get(s) and s != self.me]

    def living_wolves_count_est(self):
        dead_wolves = sum(1 for r in self.revealed.values() if r == "werewolf")
        return 2 - dead_wolves


# ---------------------------------------------------------------- helpers

def declared_votes(g):
    """Vote intentions stated in today's debate so far: {seat: count}."""
    out = {}
    for (day, rnd, seat, text) in g.speeches:
        if day != g.day:
            continue
        if WILL_VOTE.search(text) or re.search(r"\bmy vote\b|\bvote:\s*seat", text, re.I):
            for t in g.targets_in(text):
                if t != seat:
                    out[t] = out.get(t, 0) + 1
    return out


def today_votes_cast(g):
    return {t: sum(1 for (d, v, tt) in g.votes if d == g.day and tt == t)
            for t in g.name}


# ---------------------------------------------------------------- night

def decide_kill(g, options):
    # Second wolf to name: match the partner's pick so the kill lands.
    for v in (g.view.get("wolfVotes") or []):
        if v["night"] == g.day and v["seat"] == g.partner and v["target"] is not None:
            if v["target"] in options:
                return v["target"]
    score = {}
    seer_seat = g.trusted_seer() or g.sole_claimant()
    doctor_dead = any(r == "doctor" for r in g.revealed.values())
    for t in options:
        sc = 0.0
        if t == seer_seat:
            sc += 46 if doctor_dead else 30
        # A villager cleared by a credible seer can never be exiled;
        # removing them shrinks the village's safe core.
        for c, lv in g.trust.items():
            if lv in ("confirmed", "backed", "plausible"):
                chk = g.seer_claimants[c]["checks"].get(t)
                if chk and not chk[0]:
                    sc += 40
        # Strong village voices are dangerous; quiet ones less so.
        talk = sum(len(x[3]) for x in g.speeches if x[2] == t)
        sc += min(18.0, talk / 90.0)
        # Do not rescue a villager the village itself is about to exile.
        sc -= min(38.0, g.sus.get(t, 0) * 0.45)
        # A player whose loudest accusation was aimed at my partner is a
        # threat, but killing them points at the partner; a wash.
        score[t] = sc
    best = max(options, key=lambda t: (score[t], -t))
    return best


def decide_protect(g, options):
    last = (g.view.get("lastProtect") or {}).get("seat")
    seer_seat = g.trusted_seer() or g.sole_claimant()
    score = {}
    for t in options:
        sc = 0.0
        if t == seer_seat:
            sc += 100
        for c, lv in g.trust.items():
            if lv in ("confirmed", "backed"):
                chk = g.seer_claimants[c]["checks"].get(t)
                if chk and not chk[0]:
                    sc += 45  # a confirmed villager anchors the village
        if any(v == t for (_d, v, _t) in g.votes):
            sc += 4
        talk = sum(len(x[3]) for x in g.speeches if x[2] == t)
        sc += min(14.0, talk / 110.0)
        if t == last:
            sc -= 1000  # never legal twice in a row; options exclude it anyway
        score[t] = sc
    return max(options, key=lambda t: (score[t], -t))


def decide_inspect(g, options):
    mine = g.my_checks()
    order = []
    # In a claim war, settle the other claimant first.
    for t in options:
        if t in g.seer_claimants and t not in mine:
            order.append((0, t))
        elif t not in mine:
            order.append((1, -g.sus.get(t, 0), t))
    if not order:
        return None
    order.sort()
    pick = order[0][-1]
    return pick


# ---------------------------------------------------------------- speech

def my_fake_claim(g):
    """If I (as a wolf) already claimed seer, my invented history."""
    if g.me not in g.seer_claimants:
        return None
    return g.seer_claimants[g.me]["checks"]


def check_report_line(g, seat, checks):
    parts = []
    for t in sorted(checks):
        is_wolf = checks[t][0] if isinstance(checks[t], tuple) else checks[t]
        parts.append("%s: %s" % (g.ref(t), "a WEREWOLF" if is_wolf else "not a werewolf"))
    return parts


def seer_reveal_text(g, checks):
    parts = check_report_line(g, g.me, checks)
    body = "; ".join(parts)
    wolves = [t for t in checks if (checks[t][0] if isinstance(checks[t], tuple)
                                    else checks[t])]
    if wolves:
        return ("I am the seer. My checks so far: %s. We exile %s today. "
                "Doctor, please protect me tonight." % (body, g.ref(wolves[0])))[:480]
    return ("I am the seer. My checks so far: %s. No wolf found yet, but "
            "those seats are clear; we hunt among the rest today." % body)[:480]


def pick(r, *opts):
    return opts[min(len(opts) - 1, int(r[0] * len(opts)))]


def decide_speak(g, game_no, rnd):
    me = g.me
    declared = declared_votes(g)
    my_risk = declared.get(me, 0) + sum(
        1 for (d, v, t) in g.votes if d == g.day and t == me)
    sus_sorted = sorted(g.living_others(), key=lambda s: -g.sus.get(s, 0))
    top = sus_sorted[0] if sus_sorted else None

    # ---------------- I am the seer ----------------
    if g.my_role == "seer":
        mine = g.my_checks()
        found = [t for t, w in mine.items() if w and g.alive.get(t)]
        claimed = me in g.seer_claimants
        rival = [c for c in g.seer_claimants if c != me]
        framed_me = any(g.seer_claimants[c]["checks"].get(me, (False, 0))[0]
                        for c in rival)
        if found and not claimed:
            return seer_reveal_text(g, mine)
        if not claimed and (framed_me or (g.day >= 3 and mine) or my_risk >= 2):
            return seer_reveal_text(g, mine)
        if claimed:
            new = {t: mine[t] for t in mine
                   if t not in g.seer_claimants[me]["checks"]}
            if new:
                parts = check_report_line(g, me, new)
                base = "Updating my checks: %s. " % "; ".join(parts)
                if found:
                    return (base + "We exile %s today." % g.ref(found[0]))[:480]
                return (base + "Still no wolf in my checks; we hunt among "
                        "the unchecked today.")[:480]
            if found:
                return ("My result stands: %s is a werewolf. Vote them out "
                        "today; a tie only helps the wolves." % g.ref(found[0]))[:480]
            return None
        # Hidden seer: nudge without exposing the source.
        if rnd == 1 and top is not None and g.sus.get(top, 0) >= 40:
            return ("I keep coming back to %s: the voting record and the "
                    "claims do not sit right there. I want their exile "
                    "today unless something better surfaces." % g.ref(top))[:480]
        return None

    # ---------------- I am a wolf ----------------
    if g.my_role == "werewolf":
        fake = my_fake_claim(g)
        partner = g.partner
        partner_outed = partner is not None and g.sus.get(partner, 0) >= 110
        rival_claims = [c for c in g.seer_claimants if c != me]
        # Claim war on my partner: back the partner, attack the rival.
        if fake is not None:
            framed = [t for t, (w, _d) in fake.items() if w and g.alive.get(t)]
            if rnd == 1 and framed:
                return ("I am the seer and my result stands: %s is a "
                        "werewolf. Any rival claim is the wolf covering "
                        "itself. Vote %s today." % (g.ref(framed[0]),
                                                    g.ref(framed[0])))[:480]
            return None
        village_seer = g.trusted_seer() or (g.sole_claimant()
                                            if g.sole_claimant() != me else None)
        about_to_die = my_risk >= 2 or (declared.get(me, 0) >= 1 and g.day >= 3)
        wolves_left = g.living_wolves_count_est()
        living_total = sum(1 for s in g.name if g.alive.get(s))
        endgame_push = (g.day >= 3 and not rival_claims and wolves_left == 2
                        and living_total <= 6)
        if village_seer is not None and g.seer_claimants[village_seer][
                "checks"].get(me, (False, 0))[0]:
            about_to_die = True  # the real seer has me; only a counter-claim saves me
        if about_to_die or endgame_push:
            return wolf_fake_claim_text(g)
        if rnd == 1:
            # Push a villager: prefer the one the village already doubts,
            # never my partner, never a cleared seat.
            cands = [s for s in sus_sorted
                     if s != partner and g.sus.get(s, 0) > -60]
            if cands:
                t = cands[0]
                r = rng(TRIAL_SEED, game_no, g.day, "wolfpush", len(g.log))
                return pick(r,
                            "Reading the votes so far, %s looks worst to me: "
                            "their pushes keep landing on villagers. I lean "
                            "toward exiling them today." % g.ref(t),
                            "I distrust %s today: their story shifts whenever "
                            "a new claim lands. My vote is heading there."
                            % g.ref(t),
                            "Quietly, %s has added nothing and voted with "
                            "the crowd each day. That is exactly how a wolf "
                            "hides. I suspect them." % g.ref(t))[:480]
            return None
        if rnd == 2 and top is not None and top != partner:
            return ("I agree the focus stays on %s; my vote goes there "
                    "unless a claim changes things." % g.ref(top))[:480]
        return None

    # ---------------- doctor / villager ----------------
    if rnd == 1:
        ts = g.trusted_seer()
        if ts is not None:
            wolves = [t for t, (w, _d) in g.seer_claimants[ts]["checks"].items()
                      if w and g.alive.get(t)]
            if wolves:
                return ("The seer's word is the best we have: %s was "
                        "checked and is a werewolf. I am voting there "
                        "today; everyone should." % g.ref(wolves[0]))[:480]
        claims = [c for c in g.seer_claimants if g.trust.get(c) == "war"]
        if len(claims) >= 2:
            a, b = claims[0], claims[1]
            return ("Two seer claims cannot both be true: one of %s and "
                    "%s is a wolf in seer clothing. Compare their checks "
                    "with the revealed roles before voting."
                    % (g.ref(a), g.ref(b)))[:480]
        if top is not None and g.sus.get(top, 0) >= 45:
            return ("My strongest suspicion is %s: the pattern of their "
                    "votes and claims points wolf. I intend to vote there "
                    "today." % g.ref(top))[:480]
        if g.my_role == "doctor" and my_risk >= 2:
            return ("Before you exile me: I am the doctor. My protections "
                    "kept this village alive; exile me and the nights are "
                    "defenceless. Look at the vote record instead.")[:480]
        return None
    if rnd == 2:
        ts = g.trusted_seer()
        if ts is not None:
            wolves = [t for t, (w, _d) in g.seer_claimants[ts]["checks"].items()
                      if w and g.alive.get(t)]
            if wolves:
                return ("Reminder before we vote: the seer checked %s: a "
                        "werewolf. Do not split the vote."
                        % g.ref(wolves[0]))[:480]
        if top is not None and g.sus.get(top, 0) >= 45:
            return ("I am voting %s today, for the reasons already on "
                    "the table." % g.ref(top))[:480]
    return None


def wolf_fake_claim_text(g):
    """Build a consistent fake seer history and frame one villager."""
    dead_clear = [s for s, r in g.revealed.items()
                  if r != "werewolf" and not g.alive.get(s)]
    living = [s for s in g.living_others() if s != g.partner]
    frame = None
    for s in sorted(living, key=lambda s: -g.sus.get(s, 0)):
        frame = s
        break
    parts = []
    used = 0
    for s in dead_clear[:2]:
        used += 1
        parts.append("night %d I checked %s: not a werewolf" % (used, g.ref(s)))
    if frame is not None:
        parts.append("last night I checked %s: a WEREWOLF" % g.ref(frame))
    body = "; ".join(parts)
    return ("I have stayed quiet too long: I am the seer. %s. We exile "
            "%s today or the wolves take the numbers."
            % (body.capitalize(), g.ref(frame) if frame else "the wolf"))[:480]


# ---------------------------------------------------------------- vote

def decide_vote(g, options):
    me = g.me
    cast = today_votes_cast(g)
    declared = declared_votes(g)

    if g.my_role == "werewolf":
        partner = g.partner
        fake = my_fake_claim(g)
        if fake:
            framed = [t for t, (w, _d) in fake.items()
                      if w and t in options]
            if framed:
                return framed[0]
        # If the partner is certainly falling, blend in by joining.
        if partner in options and g.sus.get(partner, 0) >= 110 and \
                cast.get(partner, 0) + declared.get(partner, 0) >= 3:
            return partner
        # Ride the bandwagon against a villager; prefer the target the
        # village is already converging on.
        cands = [t for t in options if t != partner]
        if not cands:
            return None
        def wolf_score(t):
            return (cast.get(t, 0) * 10 + declared.get(t, 0) * 6
                    + max(0.0, g.sus.get(t, 0)) * 0.12
                    - (60 if t == g.trusted_seer() else 0))
        return max(cands, key=lambda t: (wolf_score(t), -t))

    # Village side.
    mine = g.my_checks() if g.my_role == "seer" else {}
    for t, w in mine.items():
        if w and t in options:
            return t  # my own confirmed wolf beats every other signal
    ts = g.trusted_seer()
    if ts is not None:
        for t, (w, _d) in g.seer_claimants[ts]["checks"].items():
            if w and t in options:
                return t
    ranked = sorted(options, key=lambda s: -g.sus.get(s, 0))
    if not ranked:
        return None
    top = ranked[0]
    top_score = g.sus.get(top, 0)
    # Coordination: if a nearly-as-suspect target is already collecting
    # votes and mine has none, join it rather than split into a tie.
    for t in ranked[1:]:
        if g.sus.get(t, 0) >= 18 and cast.get(t, 0) >= 2 and \
                cast.get(top, 0) == 0 and top_score < 200:
            return t
    if top_score >= 12:
        return top
    # Thin evidence: follow the day's strongest honest bandwagon.
    flow = {t: cast.get(t, 0) * 2 + declared.get(t, 0) for t in options}
    best_flow = max(options, key=lambda t: (flow.get(t, 0), -g.sus.get(t, 0)))
    if flow.get(best_flow, 0) >= 2 and g.sus.get(best_flow, 0) > -100:
        return best_flow
    if top_score > -40:
        return top
    return None


# ---------------------------------------------------------------- driver

def decide(game_no, view, pending):
    g = Game(view)
    kind = pending["decision"]
    options = pending.get("options") or []
    if kind == "kill":
        return {"game": game_no, "decision": "kill",
                "target": decide_kill(g, options)}
    if kind == "protect":
        return {"game": game_no, "decision": "protect",
                "target": decide_protect(g, options)}
    if kind == "inspect":
        return {"game": game_no, "decision": "inspect",
                "target": decide_inspect(g, options)}
    if kind == "vote":
        return {"game": game_no, "decision": "vote",
                "target": decide_vote(g, options)}
    if kind == "speak":
        # Which round is this? Count my speeches in today's debate.
        rnd = 1 + sum(1 for (d, r, s, _t) in g.speeches
                      if d == g.day and s == g.me)
        if rnd > 2:
            rnd = 2
        text = decide_speak(g, game_no, rnd)
        if text is None:
            return {"game": game_no, "decision": "speak", "pass": True}
        # The env folds a statement to one line and counts its length
        # after folding; fold and clamp here so an act is never refused.
        text = " ".join(text.split())
        limit = pending.get("max_chars") or 480
        if len(text) > limit:
            text = text[:limit]
        return {"game": game_no, "decision": "speak", "text": text}
    return {"game": game_no, "decision": kind, "pass": True}


def run():
    if not ENV_URL:
        return
    for _round in range(200000):
        obs = post("/observe", {})
        if obs.get("done"):
            return
        pending = sorted(obs.get("pending") or [], key=lambda p: p["game"])
        if not pending:
            continue
        for p in pending:
            try:
                act = decide(p["game"], p["view"], p)
            except Exception:
                # Never stall a game on a bug: pass/abstain safely.
                kind = p["decision"]
                if kind == "speak":
                    act = {"game": p["game"], "decision": "speak", "pass": True}
                else:
                    act = {"game": p["game"], "decision": kind, "target": None}
            try:
                post("/act", act)
            except Exception:
                pass


if __name__ == "__main__":
    run()
