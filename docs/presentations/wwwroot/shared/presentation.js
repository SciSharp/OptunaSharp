/**
 * A dependency-free shell for scrollable HTML presentations.
 * Opt in with data-slide (and optional data-notes) on sections.
 * Emits presentation:slidechange with { index, section } for deck-specific work.
 */
(() => {
  'use strict';
  const sections = [...document.querySelectorAll('[data-slide]')];
  if (!sections.length || document.querySelector('.presentation-tools')) return;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let current = 0;
  let frame = 0;

  const tools = document.createElement('nav');
  tools.className = 'presentation-tools';
  tools.setAttribute('aria-label', 'Presentation controls');
  tools.innerHTML = `
    <button type="button" data-action="previous" aria-label="Previous section" aria-keyshortcuts="PageUp">←</button>
    <span class="presentation-counter" aria-label="Current section"></span>
    <button type="button" data-action="next" aria-label="Next section" aria-keyshortcuts="PageDown">→</button>
    <button type="button" data-action="notes" aria-controls="presentation-notes" aria-expanded="false">Notes</button>
    <button type="button" data-action="fullscreen" aria-pressed="false">Fullscreen</button>
    <span class="presentation-sr-only" role="status"></span>`;
  document.body.append(tools);

  const notes = document.createElement('aside');
  notes.id = 'presentation-notes';
  notes.className = 'presentation-notes';
  notes.setAttribute('aria-label', 'Presenter notes');
  notes.hidden = true;
  notes.innerHTML = '<button type="button" aria-label="Close presenter notes">×</button><h2>Presenter notes</h2><p></p>';
  document.body.append(notes);
  const button = action => tools.querySelector(`[data-action="${action}"]`);

  function update() {
    tools.querySelector('.presentation-counter').textContent = `${current + 1} / ${sections.length}`;
    button('previous').disabled = current === 0;
    button('next').disabled = current === sections.length - 1;
    notes.querySelector('p').textContent = sections[current].dataset.notes || 'No notes for this section.';
  }

  function select(index, scroll = true) {
    const next = Math.max(0, Math.min(sections.length - 1, index));
    const changed = next !== current;
    current = next;
    if (scroll) {
      sections[current].scrollIntoView({ behavior: reducedMotion.matches ? 'instant' : 'smooth', block: 'start' });
      if (sections[current].id) history.replaceState(null, '', `#${sections[current].id}`);
    }
    update();
    if (changed) document.dispatchEvent(new CustomEvent('presentation:slidechange', {
      detail: { index: current, section: sections[current] }
    }));
  }

  function toggleNotes(show = notes.hidden) {
    notes.hidden = !show;
    button('notes').setAttribute('aria-expanded', String(show));
    if (!show && notes.contains(document.activeElement)) button('notes').focus();
  }

  tools.addEventListener('click', async event => {
    const action = event.target.closest('button')?.dataset.action;
    if (action === 'previous') select(current - 1);
    if (action === 'next') select(current + 1);
    if (action === 'notes') toggleNotes();
    if (action === 'fullscreen') {
      try {
        if (document.fullscreenElement) await document.exitFullscreen();
        else await document.documentElement.requestFullscreen();
      } catch {
        tools.querySelector('[role="status"]').textContent = 'Fullscreen is unavailable in this browser. Open the presentation in a separate browser window.';
      }
    }
  });
  notes.querySelector('button').addEventListener('click', () => toggleNotes(false));
  document.addEventListener('fullscreenchange', () => {
    const active = Boolean(document.fullscreenElement);
    button('fullscreen').textContent = active ? 'Exit fullscreen' : 'Fullscreen';
    button('fullscreen').setAttribute('aria-pressed', String(active));
  });
  if (!document.fullscreenEnabled) button('fullscreen').hidden = true;

  document.addEventListener('keydown', event => {
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === 'Escape') { toggleNotes(false); return; }
    if (event.target.closest('input, select, textarea, button, a, [contenteditable], [role="slider"], svg[tabindex], canvas[tabindex]')) return;
    if (event.key === 'PageDown' || event.key === 'ArrowRight') { event.preventDefault(); select(current + 1); }
    if (event.key === 'PageUp' || event.key === 'ArrowLeft') { event.preventDefault(); select(current - 1); }
    if (event.key.toLowerCase() === 'n') toggleNotes();
  });

  // Pick the section crossing the top quarter of the viewport. This also works for tall slides.
  function observeScroll() {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      let index = 0;
      for (let i = 0; i < sections.length; i++) {
        if (sections[i].getBoundingClientRect().top <= window.innerHeight * .25) index = i;
      }
      select(index, false);
    });
  }
  window.addEventListener('scroll', observeScroll, { passive: true });
  window.addEventListener('resize', observeScroll, { passive: true });
  update();
  observeScroll();
})();
