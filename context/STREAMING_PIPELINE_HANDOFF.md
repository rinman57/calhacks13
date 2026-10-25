# Implement incremental claim extraction and verification

## Goal and scope

Implement this change against the current repository, preserving unrelated work.
Read applicable repository instructions and inspect the latest implementation
before editing. This handoff supersedes the sequential scheduling assumptions in
`CLAUDE_IMPLEMENTATION_HANDOFF.md`; its evidence quality and validation requirements
still apply.

Reduce time until something useful is on screen. The first thing to show is the
spoken quote, marked as still being checked, as soon as that quote is validated.
Fill in the verdict on that same card when assessment finishes. Do not wait for
the complete claim list, or for every assessment, before showing a quote.

A video shorter than one chunk does not get faster from overlap. Its transcript
is a single extract, and the first card still waits on that extract plus one
claim's web research. The gain is for a longer transcript, and for any video
with more than one claim: a finished quote or verdict must not sit hidden while
a later claim is still running.

Keep the existing providers, evidence validation, and separate research and
assessment calls. No new agent framework or retrieval provider is needed.

This is incremental processing of an already available transcript, not live audio
capture. The initial caption or Cartesia wait remains out of scope. Do not
reintroduce video upload. Gemini is no longer the extractor. It was chosen for
video input, which this pipeline does not use.

## Models

- Extractor: `claude-sonnet-5-5` at effort `medium`. Its only job is a one-sentence
  summary plus exact quotes marked `claim` or `opinion`, with transcript times.
  It does not assign truth labels, confidence, explanations, or research
  questions. Python fills those `claims.json` fields so the file shape stays
  compatible: claims are `worth_checking` with `preliminary_label` `unverified`;
  opinions are not worth checking. `research_question` and `why_this_label` are
  empty. `visual_note` stays empty.
- Researcher and assessor: `claude-sonnet-5-5` at effort `medium`. The researcher
  decides what factual assertion the quote itself makes and gathers evidence.
  The assessor writes the verdict. Do not move that judgment back onto the extractor.

Do not add a model call that summarizes earlier chunks. If a chunk needs prior
context, paste a few of the preceding transcript lines and tell the extractor
not to emit claims from those lines.

## Current implementation to build on

- `extract/claims.py` sends the whole transcript to Sonnet in one request, then
  normalizes quotes and assigns `c1`, `c2`, ... after sorting the whole result.
- `extract/youtube.py` waits for that full extraction before verification.
- `extract/verify.py` uses a `ThreadPoolExecutor` with `CLAIM_WORKERS = 3`.
  Each worker researches a claim and then assesses it. Results are written
  incrementally, but verification starts from a finished `claims.json`.
- `extract/server.py` exposes jobs and builds public verdict cards.
- `extension/content.js` polls jobs. While a job is running, the panel hides
  verdicts. Cards appear only after completion.
- Tests live in `tests/test_verify.py` and `tests/test_claims_quote.py`.

Inspect these again before editing. The Sonnet extractor described above is
already the intended first pass.

## Desired flow

```text
Saved timestamped transcript
  -> chronological overlapping chunks
  -> Sonnet extraction for chunk 1
       -> validate quotes, assign stable IDs, persist claims
       -> show each quote immediately as "checking"
       -> enqueue eligible claims
  -> Sonnet extraction for chunk 2 (while research handles chunk 1)
       -> validate, deduplicate, persist, show, enqueue
  -> ...

Bounded Sonnet worker pool
  -> research the quote itself
  -> save evidence
  -> assess in a fresh conversation
  -> replace the checking card with the verdict and emit an update
```

Start claims in chronological order. Let them complete and display independently.
An early slow claim must not block a later completed result. Sort visible cards
by video timestamp. Keep a card on screen when newer results arrive, and replace
its checking state with the verdict rather than removing it.

## 1. Incremental extraction

Partition timestamped transcript segments into chronological chunks with an
initial target of about 60-90 seconds and roughly 10-15 seconds of overlap. Keep
segments intact where possible. Bound text size too, since caption segments vary.
Keep absolute timestamps. Do not offset or renumber them.

Process chunks one at a time so a Gemini-style quota spike is not replaced by a
extractor burst. A complete validated response for one chunk is enough to show its
quotes and enqueue them. Do not parse partially streamed JSON. Streaming the
model response is optional and not required.

Give each chunk the title, channel, and a few preceding transcript lines as
context. Those lines are context only. Accept a claim only when its first matched
transcript position falls in the chunk's primary interval, not in the overlap
that belongs to the previous chunk. Deduplicate the same occurrence with
transcript segment identity or character spans when available, otherwise with
normalized text plus temporal overlap. Do not merge distinct claims because they
share a timestamp or a topic. Do not drop a later repetition just because the
wording matches an earlier line.

IDs stay stable once a quote is shown and across resume. Do not renumber `c1`,
`c2` by sorting the whole file after every chunk. Persist an occurrence key tied
to the quote and its transcript position. Keep existing IDs when resuming an
artifact that already has them. Research and the panel must use that same ID.

Persist accepted claims atomically before publishing them. Record extraction
status, chunk progress, and errors. Keep the existing `statements` shape so
standalone verification still runs. A partial file must say extraction is
incomplete, and counts stay provisional until the last chunk. Resume completed
chunks from hashes of the transcript, the prompt, the model, the effort, and the
chunking rules.

Do not run a second full-transcript extraction at the end. Build the final
summary from the chunk summaries. Do not delay research for that summary.

## 2. Bounded scheduling and verification

Refactor the per-claim operation out of the fixed batch function. Keep
`python -m extract.verify` and its cache and validation behavior. Do not call
full-folder verification once per chunk. That duplicates work and races on the
files.

One coordinator per extraction job:

- A chronological bounded queue of validated eligible claims.
- A worker limit of 2-3 claims at a time.
- Queued, running, and completed occurrence IDs so a claim is not submitted twice.
- Backpressure instead of an unbounded pile of API requests.
- An extraction-finished signal and a drain condition.
- Thread-safe snapshots and one writer for the artifact files.

Each worker researches a claim, saves that evidence, then assesses it. A separate
assessor pool is not needed yet. Keep the current Sonnet settings: model
`claude-sonnet-5-5`, effort `medium`, and the existing search and fetch caps.

The researcher does the judgment the extractor skipped. It reads the exact quote, decides
what factual assertion that quote makes, and searches for support and
contradiction. It does not treat an empty `research_question` as a missing claim.
Assess useful partial evidence and record the limitations. No usable evidence is
`insufficient_evidence`, not a crash. An API failure, timeout, or cancellation
stays an execution failure. Do not drop a claim because it was difficult.

Save evidence when research ends, before assessment starts. Save each assessment
as soon as it finishes, and publish it so the matching checking card can update.

Use a deadline that starts when that claim starts work, plus a separate overall
job limit. Do not let later claims inherit a deadline that began before they
existed. Keep retries, tool use, continuations, output, and concurrency bounded.
Check whether server-tool `max_uses` resets on continuation, and enforce the
remaining per-claim budget across requests.

If one chunk fails, record the gap and continue with later chunks when that is
reasonable. Claims already queued still finish. Final status distinguishes
complete, partial, failed, and cancelled without erasing successes. If the
Anthropic key is missing, extraction still finishes and verification is skipped
with a warning.

## 3. Incremental server state

Publish structured updates through the existing job callback. Keep polling. Do
not add WebSockets or server-sent events for this pass. Do not leak API keys,
raw secrets, or unbounded fetched page text.

A job snapshot should carry, using names consistent with the current code:

- `status`: running, done, partial, error, or cancelled, as supported and documented.
- `extraction_status` and chunk counts.
- `discovered_count` and the eligible count so far.
- Per-claim states: queued, researching, assessing, complete, failed, or skipped.
- Quote cards for claims that are still checking, and verdict cards once assessed.
- Counts for queued, running, completed, and failed claims.
- A monotonically increasing revision, or an equivalent change marker.
- Stage warnings and individual errors.

The job stays running until extraction has ended and queued or running work has
drained. One `stage` string cannot describe extraction and research at the same
time. Expose activity counts or per-stage states. Do not mark an earlier stage
done only because another stage has started.

Serve polls from in-memory validated state or an atomic snapshot, not from a
file mid-write. Keep the current per-run output directory, and bind events to
the job that created them.

## 4. Incremental extension display

Render cards from running snapshots, not only from the final result. Change
`showRunning` and `showJob` so a routine poll does not hide or clear cards.
Reuse keyed cards so a one-second poll does not flicker, steal focus, or reset
an expanded detail. Keep the seek button and allow only http and https links.

As soon as a quote is validated, show it with the timestamp and a checking
state. When its verdict arrives, fill that same card. Do not wait for the
assessment to show the sentence the viewer just heard.

Progress stays provisional, for example:

```text
Reading transcript: chunk 2 of 8
5 claims found · 2 researching · 1 assessing · 1 result ready
```

Do not show a final claim total until extraction completes. When the viewer
opens another video, ignore events from the previous job. A card stays visible
while other claims continue or fail. A failed research call is not the same as
an `insufficient_evidence` verdict. Keep the current visual style.

## 5. Persistence, resume, and cache rules

- Keep compatible `claims.json`, `evidence.json`, and `verdict.json`.
- Never replace a completed row with a partial snapshot that omits it.
- Resume from finished chunks and from per-claim research and assessment caches.
- Invalidate the cache when the prompt, model, effort, transcript context, or
  chunking rules change. Keep source provenance and citation checks.
- Record timing and token or tool usage per stage. Do not reset usage on a
  continuation. Report tool counts that were actually observed.
- Keep the quote separate from interpretation. Do not treat the extractor's claim or
  opinion mark, or the placeholder `unverified` label, as evidence.

Do not bundle unrelated prompt-policy changes, model swaps, or PDF extraction
into this scheduling work. The Sonnet extractor and the Sonnet research handoff
described above are already the base. Report any other limitation separately.

## 6. Tests and measurable acceptance criteria

Use mocked extraction and verification calls. Do not spend API credits on these
tests.

1. Block chunk 2. Chunk 1's quote is published, and its research starts, before
   chunk 2 returns.
2. Block extraction completion. A checking card, and then an assessed card, is
   present in a running job snapshot and can be rendered before final status.
3. Make claim 1 slow and claim 2 fast. Claim 2 appears first.
4. Overlapping chunks do not duplicate one occurrence, change a published ID, or
   merge unrelated assertions. Cover a claim that crosses a chunk boundary.
5. Concurrency and queue size stay bounded. Callbacks, retries, and resume do
   not research the same claim twice. Deadlines and continuations cannot loop.
6. One chunk or claim failure keeps the successes, reports the gap, and drains
   the workers. A video with no claims ends normally.
7. Resume reuses matching work, drops stale work when inputs change, and keeps IDs.
8. Polling and navigation ignore a stale job. Results survive the next poll.
9. Existing quote, evidence, citation, and standalone CLI tests still pass.
10. The extractor schema sent to Sonnet does not ask for a truth label, confidence,
    or research question. Saved claims still contain those fields, filled without
    a model judgment.

Record monotonic durations for:

- Job start to transcript ready.
- Transcript ready to the first validated quote.
- First quote to the first research start.
- Job start, and transcript ready, to the first visible quote and to the first
  visible assessment. Keep those two metrics separate.
- Total duration and per-stage durations.

An assessment is not a failure notification. Measure when the server publishes a
card separately from when the browser paints it, if both are measured. Compare
the old full-transcript path with the incremental path on the same saved
transcript and mocked delays. Do not claim a speedup from a diagram.

## Delivery

Implement the feature end to end, run the checks, and update the README with
configuration, resume behavior, provisional progress, and limitations. A live
smoke test must stay small and report actual usage. Do not run a broad paid
benchmark. If credentials are missing, finish the mocked tests and say that live
behavior was not verified.

Explain the changes, the tests, the measured time to the first quote and the
first assessment, and the remaining limits. The target is earlier output on
screen, not necessarily a shorter total job or a lower bill. Overlapping chunks
can increase input tokens, and a chunk has less context than the full transcript.
Document both.
