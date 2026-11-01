"""Exercise sidebar/selected-video rendering with a small offline DOM stub."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import maus_export.webserve_fold as webserve_fold


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_progress_is_scoped_to_selected_video(tmp_path: Path) -> None:
    html = (
        Path(webserve_fold.__file__)
        .with_name("fold_dashboard.html")
        .read_text()
    )
    assert html.index('<video id="video"') < html.index('<div id="progress"')
    assert "Pipeline processing</h2>" not in html
    script = re.search(r"<script>([\s\S]*?)</script>", html).group(1)
    script = script.rsplit("pollProgress();", 1)[0]
    bootstrap = r"""
const assert = require('node:assert/strict');
class Element {
  constructor(tag) {
    this.tag = tag; this.children = []; this.dataset = {}; this.style = {};
    this.attrs = {}; this.classes = new Set(); this._text = '';
    this.classList = {toggle: (name, on) => on ?
      this.classes.add(name) : this.classes.delete(name)};
  }
  set textContent(value) { this._text = value; this.children = []; }
  get textContent() {
    return this._text + this.children.map(child => child.textContent).join('');
  }
  append(...items) { for (const item of items) this.insertBefore(item, null); }
  insertBefore(item, before) {
    if (item.parent)
      item.parent.children.splice(item.parent.children.indexOf(item), 1);
    const index = before ? this.children.indexOf(before) : this.children.length;
    this.children.splice(index, 0, item); item.parent = this;
  }
  replaceChildren(...items) {
    this.children.forEach(item => item.parent = null);
    this.children = []; this._text = ''; this.append(...items);
  }
  setAttribute(key, value) { this.attrs[key] = value; }
  removeAttribute(key) {
    delete this.attrs[key]; if (key === 'src') this.src = '';
  }
  remove() {
    this.parent.children.splice(this.parent.children.indexOf(this), 1);
    this.parent = null;
  }
  load() {}
  addEventListener() {}
  getBoundingClientRect() { return {width: 600, height: 210}; }
  getContext() { return {scale() {}}; }
}
const elements = new Map();
const document = {
  createElement: tag => new Element(tag),
  querySelector: key => {
    if (!elements.has(key)) elements.set(key, new Element(key));
    return elements.get(key);
  },
  querySelectorAll: () => [...runButtons.values()]
};
const window = {addEventListener() {}};
const devicePixelRatio = 1;
const setTimeout = () => {};
"""
    assertions = r"""
(async () => {
  progressJobs = new Map(Array.from({length: 500}, (_, index) => {
    const id = index + 1;
    return [id, {id, state: id === 2 ? 'running' : 'completed', fraction: .5,
      frames_done: 10, frames_total: 20, fps: 5, seconds_per_frame: .2,
      elapsed_seconds: 2, eta_seconds: 2, updated_at: 1}];
  }));
  cachedRuns = new Map([[1, {id: 1, events: [], dispense_ok: true}]]);
  selectedId = 1;
  renderSidebar(); renderProgress();
  const sidebar = document.querySelector('#runs');
  assert.equal(sidebar.children.length, 500);
  const first = runButtons.get(1);
  assert.equal(first.children[0].children.length, 1); // completed: no spinner
  assert.equal(runButtons.get(2).children[0].children[1].className, 'spinner');
  renderSidebar();
  assert.equal(runButtons.get(1), first); // preserve button and focus
  assert.equal(sidebar.children[0], first);
  const panel = document.querySelector('#progress');
  assert.equal(panel.children.filter(el => el.tag === 'progress').length, 1);
  assert.equal(panel.children[1].attrs['aria-label'],
    'Run 1 processing completion');
  await showRun(2); // active run is selectable before artifacts exist
  assert.equal(selectedId, 2);
  assert.equal(selected, null);
  assert.equal(video.src, '');
  assert.equal(panel.children[1].attrs['aria-label'],
    'Run 2 processing completion');
  assert.match(document.querySelector('#events').textContent,
    /after processing/);
  assert.equal(runButtons.get(2).classes.has('active'), true);
  progressJobs.get(2).stale = true;
  renderSidebar();
  assert.equal(runButtons.get(2).children[0].children[1].textContent, '!');
  // Failed jobs leave the regular list and explain the exception.
  const job = progressJobs.get(2);
  job.state = 'failed'; job.stale = false;
  job.error = 'FileNotFoundError'; job.error_message = 'Video not found';
  renderSidebar(); renderProgress();
  const failedSection = document.querySelector('#failed-section');
  const failedList = document.querySelector('#failed-runs');
  assert.equal(sidebar.children.length, 499);
  assert.equal(failedList.children.length, 1);
  assert.equal(failedSection.hidden, false);
  assert.equal(document.querySelector('#failed-summary').textContent,
    'Failed runs (1)');
  assert.match(failedList.textContent, /FileNotFoundError: Video not found/);
  assert.match(panel.textContent, /FileNotFoundError: Video not found/);
  failedSection.open = true;
  renderSidebar();
  assert.equal(failedSection.open, true); // polling preserves expansion
  delete job.error_message;
  renderSidebar();
  assert.match(failedList.textContent, /no detailed message recorded/);
  job.state = 'running';
  renderSidebar();
  assert.equal(failedSection.hidden, true);
  assert.equal(failedList.children.length, 0);
  assert.equal(sidebar.children.length, 500);
  // Completion refreshes the run list and loads the selected video's output.
  const completed = {...progressJobs.get(2), stale: false,
    state: 'completed', updated_at: 2};
  global.fetch = async path => ({ok: true, json: async () => {
    if (path === '/api/progress') return [completed];
    if (path === '/api/runs') return [{id: 2, events: [], dispense_ok: null}];
    return {id: 2, timeline: [], events: []};
  }});
  await pollProgress();
  assert.equal(selected.id, 2);
  assert.equal(sidebar.children.length, 1);
  assert.equal(video.src, '/video/2');
  assert.match(panel.textContent, /completed/);
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    path = tmp_path / "dashboard-ui.js"
    path.write_text(bootstrap + script + assertions)
    result = subprocess.run(
        [shutil.which("node"), str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
