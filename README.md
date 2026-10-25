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
  audio.m4a               audio file, only when no usable captions are available
  claims.json             exact quotes worth exploring
  evidence.json           sources the researcher actually retrieved
  verdict.json            the assessor's verdict for each researched claim
  extraction_report.json  which files were written, plus any warnings
```

The folder timestamp is the local Pacific time when that extract started. The label is `PDT` during daylight-saving time and `PST` otherwise.

Haiku keeps a quote only when it would matter if it were false. It reads the channel, the title, and the first paragraph of the description, so a news or explainer video is treated differently from a challenge, comedy, or vlog. It skips jokes, the speaker's own experience, fragments, talk-timing asides, and the plot or scoreboard of an entertainment video. A product, health, or public-money fact inside that kind of video can still be kept. Haiku does not decide whether a kept claim is true. On a long transcript it reads chronological chunks of about 75 seconds, and each checkable quote is sent to research as soon as that chunk is saved. Only statements saved as `kind` `claim` are researched.

Transcript segments keep the caption wording and times.

Caption lookup tries two sources, in order:

1. `youtube-transcript-api`, which prefers a manual caption track over automatic captions.
2. yt-dlp's caption download (`json3`, then WebVTT) if the first source fails.

A track with no spoken words, such as a WebVTT header, is skipped. If every track is unusable, or the caption download fails, the pipeline downloads audio and transcribes it with Cartesia `ink-whisper`. That key is `CARTESIA_API_KEY` in `.env`. Metadata still lists the caption languages YouTube advertises when the caption text itself cannot be downloaded.

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
5. A small icon sits at the top right. Click it to open a sidebar, then choose **Extract**. The extension button can start the same check. After an extension update, click the reload icon on `chrome://extensions`, then refresh the YouTube tab.

The sidebar stays closed until that icon is clicked. A status line names the current stage, and rows appear as verdicts finish. Each row is a timestamp and a verdict. Click a timestamp to seek the player. Click the row for the quote, a one or two sentence explanation, the sources, and a longer explanation. Claims marked `supported` stay in `verdict.json` and are left out of the list. Finished files land in `data/<video_id>_<timestamp>/`. **Run this on each video played in this tab** is on unless it has been turned off, and it starts a new extract when the watch URL changes in that tab.

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

Put the API keys in `.env` in the project folder. `ANTHROPIC_API_KEY` is used for claim extraction, research, and assessment. `CARTESIA_API_KEY` is used only when no usable captions are available. `.env` is gitignored.

## Research and assessment

After a checkable quote is saved, a coordinator gives it to a research worker. Up to three claims are researched at once. When research for a claim finishes, a shared assessor writes the verdict. One assessor covers four claims: one assessor for 1–4 claims, two for 5–8, three for 9–12, and four from 13 on. Both steps use Claude Sonnet 5.5. The model id is `claude-sonnet-5-5`. Set `ANTHROPIC_MODEL` to use a different id.

1. The researcher can search the web and fetch pages, at most 5 searches and 5 fetches for a claim. Python writes `evidence.json` from the tool results it observed. Search snippets stay labeled as snippets. A passage is `fetched_text` only when its quotation is found in the fetched page.
2. A separate assessor call, with no web access, reads that saved evidence and writes the verdict. `verdict.json` is one of `supported`, `contradicted`, `misleading`, `insufficient_evidence`, or `not_checkable`.

`python -m extract.verify data/<video folder>` reuses a finished claim when the claim text, transcript context, model, and evidence are unchanged. `--refresh` runs the work again. `--limit 1` caps how many checkable claims are sent, which keeps a trial small. A missing Anthropic key skips this stage and leaves the saved transcript in place.

These verdicts are automated assessments, not definitive proof. Web fetch does not run JavaScript, so some pages come back empty. The researcher can miss sources, and a demand spike or a cutoff is recorded as a failed or partial research status instead of being turned into `insufficient_evidence`.

Use this on videos the team is allowed to process. The page panel and the command save the transcript, and the audio only when no usable captions are available. YouTube's developer policies restrict storing audiovisual content for API clients. The official captions API also only downloads captions for videos the caller can edit, so this step uses public page data instead of that API.

## When captions fail

If YouTube lists captions but the text cannot be downloaded, the folder still gets metadata. `caption_languages` lists the advertised tracks, and `transcript_error` records the refusal. A missing caption file does not mean the video has no captions. The pipeline then downloads audio and asks Cartesia for a transcript. It does not download the picture.

## What is deferred

- A marker drawn on the YouTube timeline. Verdict timestamps seek the player, but they are not yet pins on the scrubber.
- Live capture of only the audio that has already played. The page panel requests the full public transcript for the open URL.
