# calhacks13

YouTube claim companion for Cal Hacks. The product idea, from the [shared planning thread](https://chatgpt.com/s/cx_6adc5c157b8c8191b8867720e12bcee0), is to watch an explanatory video, find factual claims, and attach evidence to the timestamp where each claim was said.

This repository currently does the first step only: given a public YouTube URL, save the video, the audio, and a timestamped transcript when YouTube will return one. Claim extraction, evidence search, and on-screen markers come later.

## What this step saves

`python -m extract` writes one folder per video:

```text
data/<video_id>/
  metadata.json           title, channel, duration, caption languages
  transcript.json         timestamped segments, when captions can be fetched
  transcript.txt          the same segments as [mm:ss.mmm] lines
  audio.m4a               audio-only file for a later speech-to-text fallback
  video.mp4               one playable file with picture and sound
  extraction_report.json  which files were written, plus any warnings
```

Transcript segments keep the caption wording and times. Nothing in this step decides whether a statement is true.

Caption lookup tries two sources, in order:

1. `youtube-transcript-api`, which prefers a manual caption track over automatic captions.
2. yt-dlp's caption download (`json3`, then WebVTT) if the first source fails.

Metadata still lists the caption languages YouTube advertises when the caption text itself cannot be downloaded.

## Try it on a YouTube page

The page panel extracts the video that is open. It downloads the full file from that URL, so playback can keep going and the viewer does not have to finish the video first.

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

The panel shows each stage: details, captions, audio, then video. Finished files land in `data/<video_id>/`. Checking **Also extract the next videos in this tab** runs the same extraction when the watch URL changes in that tab.

The command-line extractor is the same work without the page:

```bash
python -m extract "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Setup

Node is required. yt-dlp uses it to solve YouTube's player challenge. Without Node, the single combined video file was missing and only separate video and audio streams were offered.

ffmpeg is optional. Without it, the video stays on the single combined format YouTube still serves, which is usually 360p. A higher resolution is a separate video stream plus a separate audio stream, and ffmpeg is what merges them.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m extract "https://www.youtube.com/watch?v=VIDEO_ID"
```

Useful flags:

```bash
python -m extract "https://youtu.be/VIDEO_ID" --out data --languages en
python -m extract VIDEO_ID --no-video
```

`--no-video` still saves metadata, audio, and captions. Downloaded media stays in `data/` and is gitignored.

Use this on videos the team is allowed to process. The page panel and the command both download the video from its URL. YouTube's developer policies restrict storing audiovisual content for API clients. The official captions API also only downloads captions for videos the caller can edit, so this step uses public page data instead of that API.

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

- Speech-to-text on `audio.m4a` when caption download is blocked. That is the planned fallback, and this run is the case it is for.
- Reading frames from `video.mp4`.
- Turning transcript lines into claims.
- Searching for evidence and attaching a marker to a timestamp.
- Live tab capture of only the audio and frames that have already played. The page panel downloads the whole video from its URL instead.
