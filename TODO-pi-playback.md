# Low-bandwidth Pi playback (#193)

- [x] Measure original size, codec and LAN transfer rate.
- [x] Add a separate H.264/yuv420p fast-start playback artifact.
- [x] Keep original analysis and timing outputs unchanged.
- [x] Validate proxy provenance and fall back on missing/stale output.
- [x] Compress graph/HTML responses when gzip is accepted.
- [x] Support suffix byte ranges and preserve original-video access.
- [x] Deploy the current private example and restart the website service.
- [x] Final tests and documentation.
- [ ] Merge stacked changes after #192 and required CI checks.

130 fold tests passed, 3 skipped; lint, formatting and docs checks passed.

## Measurements (2026-11-01)

- Original: 11,983,439 bytes, MPEG-4 Part 2, 900x900, 1.73 Mbps.
- Initial 256 KiB LAN sample: approximately 23 KB/s.
- Playback: 833,753 bytes, H.264, 480x480, approximately 120 kbps;
  duration unchanged at 55.4 seconds. About 14 times smaller.
- Graph JSON: 303,006 bytes uncompressed, 26,789 bytes with gzip.
- Post-deployment LAN sample: approximately 168 KB/s; Wi-Fi throughput is
  variable, so this is not proof of a network-level speed improvement.
- Proxy byte ranges and the complete compressed 554-row graph were verified.
- A low-bitrate preview loses fine detail. Original downloads remain available.
- Encoding is separate from inference; no Raspberry Pi timing claim changed.
- All video and per-run artifacts stay in private data directories.
