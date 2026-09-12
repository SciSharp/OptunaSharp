import { sourceLines, findSourceMatches } from './model.js';

const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const tokenPattern = /(\/\/.*$|\/\*.*?(?:\*\/|$)|@?"(?:""|\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:public|private|protected|internal|static|readonly|class|sealed|record|override|virtual|abstract|new|return|if|else|for|foreach|while|break|continue|using|namespace|var|void|int|double|bool|string|null|true|false|throw|try|catch|finally|out|ref|in|is|not|as|switch|case|default|params|base|this|typeof|nameof)\b|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)/g;
function colorize(line) {
  let output = '', start = 0;
  for (const match of line.matchAll(tokenPattern)) {
    output += escape(line.slice(start, match.index));
    const token = match[0];
    const category = token.startsWith('//') || token.startsWith('/*') ? 'comment'
      : /^["'@]/.test(token) ? 'string' : /^\d/.test(token) ? 'number' : 'keyword';
    output += `<span class="token-${category}">${escape(token)}</span>`;
    start = match.index + token.length;
  }
  return output + escape(line.slice(start));
}

/** A complete source view: stage highlights never remove surrounding source lines. */
export class CodeView {
  constructor(bundle, onLocation) {
    this.bundle = bundle;
    this.onLocation = onLocation;
    this.file = null;
    this.selectedLine = 1;
    this.matches = [];
    this.matchIndex = -1;
    this.stageRange = null;
    this.container = document.getElementById('source-code');
    this.fileSelect = document.getElementById('source-file');
    this.symbolSelect = document.getElementById('source-symbol');
    this.searchInput = document.getElementById('source-search');
    this.fileSelect.replaceChildren(...bundle.files.map(file => new Option(`${file.name} · ${file.lineCount} lines`, file.id)));
    this.fileSelect.addEventListener('change', () => this.showFile(this.fileSelect.value, 1));
    this.symbolSelect.addEventListener('change', () => {
      const symbol = this.file.symbols.find(s => s.id === this.symbolSelect.value);
      if (symbol) this.showSymbol(this.file.id, symbol.id);
    });
    this.searchInput.addEventListener('input', () => this.search());
    this.searchInput.addEventListener('keydown', event => {
      if (event.key === 'Enter') { event.preventDefault(); this.nextMatch(); }
    });
    document.getElementById('source-next-match').addEventListener('click', () => this.nextMatch());
    this.container.addEventListener('click', event => {
      const row = event.target.closest('[data-line]');
      if (row) this.selectLine(Number(row.dataset.line), false);
    });
    this.container.addEventListener('keydown', event => {
      const key = event.key;
      if (!['ArrowUp', 'ArrowDown', 'Home', 'End'].includes(key)) return;
      event.preventDefault(); event.stopPropagation();
      const line = key === 'Home' ? 1 : key === 'End' ? this.lines.length : this.selectedLine + (key === 'ArrowUp' ? -1 : 1);
      this.selectLine(line);
    });
  }

  showFile(fileId, line = 1, range = null) {
    const file = this.bundle.files.find(f => f.id === fileId);
    if (!file) return;
    const changed = this.file?.id !== fileId;
    this.file = file;
    this.stageRange = range;
    this.fileSelect.value = fileId;
    if (changed) {
      this.lines = sourceLines(file.content);
      this.container.innerHTML = this.lines.map((text, i) => `<div class="source-line" data-line="${i + 1}" id="code-L${i + 1}"><button type="button" class="line-number" tabindex="-1" aria-label="Select line ${i + 1}">${i + 1}</button><span class="line-code">${colorize(text)}</span></div>`).join('');
      this.symbolSelect.replaceChildren(new Option('Jump to a function…', ''), ...file.symbols.map(s => new Option(`${s.name} · L${s.startLine}`, s.id)));
      document.getElementById('source-breadcrumb').textContent = file.path;
      document.getElementById('source-provenance').textContent = `Complete file · ${file.lineCount} lines · SHA-256 ${file.sha256.slice(0, 16)}…`;
      this.container.setAttribute('aria-label', `Complete source of ${file.name}, ${file.lineCount} lines`);
      this.search(false);
    }
    this.container.querySelectorAll('.is-active').forEach(el => el.classList.remove('is-active'));
    if (range) for (let n = range.startLine; n <= range.endLine; n++) {
      this.container.querySelector(`[data-line="${n}"]`)?.classList.add('is-active');
    }
    this.selectLine(line);
  }

  showSymbol(fileId, symbolId, range = true) {
    const file = this.bundle.files.find(f => f.id === fileId);
    const symbol = file?.symbols.find(s => s.id === symbolId);
    if (symbol) this.showFile(fileId, symbol.startLine, range ? symbol : null);
  }

  selectLine(line, scroll = true) {
    if (!this.file) return;
    this.selectedLine = Math.max(1, Math.min(this.lines.length, line));
    const old = this.container.querySelector('.is-selected');
    old?.classList.remove('is-selected');
    if (old) old.querySelector('.line-number').tabIndex = -1;
    const row = this.container.querySelector(`[data-line="${this.selectedLine}"]`);
    row?.classList.add('is-selected');
    if (row) row.querySelector('.line-number').tabIndex = 0;
    if (scroll && row) this.container.scrollTop = Math.max(0, row.offsetTop - this.container.offsetTop - this.container.clientHeight * .3);
    const symbols = this.file.symbols.filter(s => s.startLine <= this.selectedLine && s.endLine >= this.selectedLine);
    const symbol = symbols.sort((a, b) => (a.endLine - a.startLine) - (b.endLine - b.startLine))[0];
    this.symbolSelect.value = symbol?.id || '';
    document.getElementById('symbol-description').textContent = symbol
      ? `${symbol.name} · lines ${symbol.startLine}–${symbol.endLine}${symbol.summary ? ` · ${symbol.summary}` : ''}`
      : `Line ${this.selectedLine} · declaration, documentation, or file context`;
    this.renderReferences(symbol);
    const provenance = document.getElementById('source-provenance');
    let sourcebookLink = provenance.querySelector('a');
    if (!sourcebookLink) {
      sourcebookLink = document.createElement('a');
      sourcebookLink.target = '_blank';
      sourcebookLink.rel = 'noopener';
      sourcebookLink.textContent = 'Open this line in the sourcebook ↗';
      provenance.append(document.createElement('br'), sourcebookLink);
    }
    sourcebookLink.href = `/nsga3-atlas/sourcebook.html#${this.file.id}-L${this.selectedLine}`;
    this.onLocation?.({ fileId: this.file.id, line: this.selectedLine, symbol });
    this.updateCounter();
  }

  renderReferences(symbol) {
    const host = document.getElementById('cross-references');
    host.replaceChildren();
    const calls = this.bundle.calls || [];
    const related = symbol ? calls.filter(c => c.from === symbol.id || c.to === symbol.id) : [];
    if (!related.length) { host.textContent = 'Select a mapped function to inspect local call references.'; return; }
    for (const call of related) {
      const outgoing = call.from === symbol.id;
      const id = outgoing ? call.to : call.from;
      const file = this.bundle.files.find(f => f.symbols.some(s => s.id === id));
      const target = file?.symbols.find(s => s.id === id);
      if (!target) continue;
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'xref-link';
      button.textContent = `${outgoing ? 'calls →' : 'called by ←'} ${target.name} · L${target.startLine}`;
      button.addEventListener('click', () => this.showSymbol(file.id, id));
      host.append(button);
    }
  }

  search(scroll = true) {
    if (!this.file) return;
    this.matches = findSourceMatches(this.lines, this.searchInput.value);
    this.matchIndex = -1;
    this.container.querySelectorAll('.is-search-match').forEach(el => el.classList.remove('is-search-match'));
    for (const n of this.matches) this.container.querySelector(`[data-line="${n}"]`)?.classList.add('is-search-match');
    if (scroll && this.matches.length) this.nextMatch();
    else this.updateCounter();
  }
  nextMatch() {
    if (!this.matches.length) return;
    this.matchIndex = (this.matchIndex + 1) % this.matches.length;
    this.selectLine(this.matches[this.matchIndex]);
  }
  updateCounter() {
    document.getElementById('source-counter').textContent = this.searchInput.value
      ? `${this.matches.length ? this.matchIndex + 1 : 0} / ${this.matches.length} matching lines`
      : `L${this.selectedLine} / ${this.file?.lineCount || 0} · full source`;
  }
}
