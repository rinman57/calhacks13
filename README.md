# calhacks13

YouTube claim companion for Cal Hacks. The product idea, from the [shared planning thread](https://chatgpt.com/s/cx_6adc5c157b8c8191b8867720e12bcee0), is to watch an explanatory video, find factual claims, and attach evidence to the timestamp where each claim was said.

Given a public YouTube URL, the pipeline saves a timestamped transcript, asks Claude Haiku 5.5 for a fast quote list, then asks Claude Sonnet 5.5 to research and assess the checkable claims. It does not download the picture.

## What this step saves

`python -m extract` writes one folder per video:

```text
data/<video_id>_<YYYY-MM-DD>_<HH-MM-SS>_<PDT or PST>/
  metadata.json           title, channel, duration, caption languages
  transcript.json         timestamped segments, when captions or Cartesia succeed
  transcript.txt          the same segments as [mm:ss.mmm] lines
  audio.m4a               audio file, only when captions are missing
  claims.json             important, relevant factual quotes selected for research
  evidence.json           sources the researcher actually retrieved
  verdict.json            the assessor's verdict for each researched claim
  extraction_report.json  which files were written, plus any warnings
```

The folder timestamp is the local Pacific time when that extract started. October is daylight-saving time, so those folders are labeled `PDT`.

Haiku reads the full transcript together with the title, channel, and description. It uses surrounding dialogue, attribution, setup/punchline, and nearby rebuttals to distinguish serious factual assertions from jokes, opinions, sarcasm, satire, and personal remarks. It selects only complete, independently checkable claims whose accuracy would materially affect the viewer's understanding or decisions. Stunt inventory, scores, ordinary trivia, and unclear fragments are omitted. It does not decide truth. Python discards non-claim candidates and anything without an explicit boolean `worth_checking: true`; rejected candidates do not enter `claims.json` or research. Empty output is valid.

The selection prompt uses a three-way rubric inspired by [ClaimBuster's research](https://idir.uta.edu/~naeemul/file/claimbuster-cikm15-hassan-poster.pdf): nonfactual speech, unimportant facts, and check-worthy facts. Only the last group qualifies. It explicitly rejects isolated arithmetic and arbitrary statistics used as filler, requires a concrete material consequence, and permits whole videos with no candidates. Applied calculations with real consequences remain eligible. This is a prompt adaptation, not a ClaimBuster API integration or a validated reproduction of its classifier.

Python now also requires the extractor's explicit `checkworthiness: important_fact` classification, and blocks standalone multiplication filler even when the model incorrectly admits it. This narrow rule does not exclude applied calculations with units or consequences. New claim files record `selection_version: checkworthiness-3`. After changing Python code, stop and restart the server: a running process keeps imported code in memory. The startup message and `/health` endpoint report the active selection version.

Both research and assessment recheck context. Research is instructed to avoid web tools for ineligible remarks; assessment must not label nonliteral speech or uncertainty as misinformation. Transcript text cannot always establish sarcasm or vocal tone, so this remains a model judgment, not a guarantee. Only completed `contradicted` or `misleading` results for eligible claims appear as public flags. Other results remain in the saved files and execution problems remain in warnings. No flags does not mean all statements have been verified.

The existing three-worker limit is unchanged and selection adds no model call. Fewer selected claims should reduce research work, but a run can still exceed a minute: full-transcript extraction precedes research, and each claim requires web retrieval followed by a separate assessment. No latency improvement has been measured for this change. Old saved claim lists retain their old selection decisions; rerun extraction to apply the new policy.

Transcript segments keep the caption wording and times. Nothing in this step decides whether a statement is true.

Caption lookup tries two sources, in order:

1. `youtube-transcript-api`, which prefers a manual caption track over automatic captions.
2. yt-dlp's caption download (`json3`, then WebVTT) if the first source fails.

Metadata still lists the caption languages YouTube advertises when the caption text itself cannot be downloaded.

## Try it on a YouTube page

The page panel extracts the video that is open. It saves the transcript from that URL, so playback can keep going and the viewer does not have to finish the video first.

Start the local extractor from the project folder:

```bash
source .venv/bin/activate
python -m extract.server
```

Leave that process running. Then in Chrome:

1. Open `chrome://extensions`.
2. Turn on Developer mode.
3. Choose Load unpacked and select the `extension` folder in this repo.
4. Open a YouTube watch page. Reload the tab if it was already open.
5. Use the **Extract this video** panel at the bottom right, or the extension button. After an extension update, click the reload icon on `chrome://extensions` before trying the button again.

The panel shows each stage: details, captions, then claims. Audio is saved only when the captions are missing. Finished files land in `data/<video_id>_<timestamp>/`. Checking **Run this on each video played in this tab** runs the same extraction when the watch URL changes in that tab.

The command-line extractor is the same work without the page:

```bash
python -m extract "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Setup

Node is required. yt-dlp uses it to solve YouTube's player challenge for captions and, when captions are missing, the audio download.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m extract "https://www.youtube.com/watch?v=VIDEO_ID"
```

Useful flags:

```bash
python -m extract "https://youtu.be/VIDEO_ID" --out data --languages en
python -m extract.verify data/<video folder> --limit 1
```

An English request also accepts a regional track such as `en-US`, so a listed caption is used before speech-to-text. Saved media stays in `data/` and is gitignored.

## Research and assessment

After `claims.json` exists, each checkable claim gets two Claude Sonnet 5.5 calls: research, then assessment. Up to three claims run at the same time. The search and fetch limits are unchanged. The model id is `claude-sonnet-5-5`. Set `ANTHROPIC_MODEL` to use a different id. The key is `ANTHROPIC_API_KEY` in `.env`, which is gitignored.

1. The researcher can search the web and fetch pages, at most 5 searches and 5 fetches for a claim. Python writes `evidence.json` from the tool results it observed. Search snippets stay labeled as snippets. A passage is `fetched_text` only when its quotation is found in the fetched page.
2. The assessor starts a fresh conversation with no web tools. It sees the claim, nearby transcript lines, and the saved evidence. `verdict.json` is one of `supported`, `contradicted`, `misleading`, `insufficient_evidence`, or `not_checkable`.

`python -m extract.verify data/<video folder>` reuses a finished claim when the claim text, transcript context, model, and evidence are unchanged. `--refresh` runs the work again. `--limit 1` caps how many checkable claims are sent, which keeps a trial small. A missing Anthropic key skips this stage and leaves the saved transcript in place.

These verdicts are automated assessments, not definitive proof. Web fetch does not run JavaScript, so some pages come back empty. The researcher can miss sources, and a demand spike or a cutoff is recorded as a failed or partial research status instead of being turned into `insufficient_evidence`.

The page panel shows the verdict, the explanation, limitations, and source links. A timestamp in a verdict seeks the YouTube player. Reload the extension on `chrome://extensions` after pulling these panel changes, then refresh the YouTube tab.

Use this on videos the team is allowed to process. The page panel and the command save the transcript, and the audio only when captions are missing. YouTube's developer policies restrict storing audiovisual content for API clients. The official captions API also only downloads captions for videos the caller can edit, so this step uses public page data instead of that API.

## What a real run showed

Checked on 24 Oct 2026 with [Me at the zoo](https://www.youtube.com/watch?v=jNQXAC9IVRw) (`jNQXAC9IVRw`, 19 seconds, channel `jawed`).

| File | Result |
| --- | --- |
| `metadata.json` | Title, channel, duration 19s, public availability |
| `audio.m4a` | 302 KB audio-only M4A |
| `video.mp4` | 614 KB MP4. The file contains both a video track and an audio track |
| Captions | Listed, then blocked. See below |

YouTube reported manual captions in `en` and `de`, and automatic captions in `en` and `de`. The text never downloaded:

- `youtube-transcript-api` returned `IpBlocked`.
- yt-dlp's caption request returned HTTP 429 from `youtube.com/api/timedtext`.

Media downloads from `googlevideo.com` succeeded on the same run. So a missing transcript here means the caption endpoint refused the request. It does not mean the video has no captions. The report records that distinction in `caption_languages` and `transcript_error`.

Before Node and the `yt-dlp-ejs` solver were enabled, yt-dlp only offered separate video and audio streams for this upload, and the combined MP4 request failed. After the solver ran, the combined MP4 downloaded.

ffmpeg is not installed on this machine, so the check stopped at that combined file.

## What is deferred

- A marker drawn on the YouTube timeline. Verdict timestamps seek the player, but they are not yet pins on the scrubber.
- Live tab capture of only the audio and frames that have already played. The page panel downloads the whole video from its URL instead.
