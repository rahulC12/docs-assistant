/* Docs Assistant — front end.
 *
 * No framework and no build step: the whole interface is three static
 * files served by FastAPI. That is a deliberate choice for a tool people
 * should be able to clone and run in one command.
 */

const $ = (id) => document.getElementById(id);

const els = {
  drop:      $("drop"),
  fileInput: $("file-input"),
  docs:      $("docs"),
  railEmpty: $("rail-empty"),
  chunkChip: $("chunk-count"),
  modeDot:   $("mode-dot"),
  modeText:  $("mode-text"),
  thread:    $("thread"),
  opening:   $("opening"),
  examples:  $("examples"),
  form:      $("form"),
  q:         $("q"),
  send:      $("send"),
  note:      $("composer-note"),
};

let library = { documents: [], chunk_count: 0 };
let busy = false;

/* ─────────────────────────── helpers ─────────────────────────── */

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function note(message, isError = false) {
  els.note.textContent = message;
  els.note.classList.toggle("is-error", isError);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch { /* keep */ }
    throw new Error(detail);
  }
  return response.json();
}

/* ─────────────────────────── library ─────────────────────────── */

function renderLibrary() {
  const { documents, chunk_count } = library;

  els.railEmpty.hidden = documents.length > 0;
  els.chunkChip.hidden = chunk_count === 0;
  els.chunkChip.textContent = `${chunk_count} passages`;

  els.docs.innerHTML = documents.map((doc) => `
    <li class="doc" data-name="${esc(doc.filename)}">
      <span class="doc__spine"></span>
      <span class="doc__body">
        <span class="doc__name" title="${esc(doc.filename)}">${esc(doc.filename)}</span>
        <span class="doc__meta">${plural(doc.pages, "page")} · ${plural(doc.chunks, "passage")}</span>
      </span>
      <button class="doc__remove" data-id="${esc(doc.doc_id)}"
              aria-label="Remove ${esc(doc.filename)}">×</button>
    </li>`).join("");

  els.docs.querySelectorAll(".doc__remove").forEach((button) => {
    button.onclick = () => removeDoc(button.dataset.id);
  });

  const ready = documents.length > 0;
  els.send.disabled = !ready || busy;
  els.q.disabled = !ready;
  if (!busy) note(ready ? "" : "Add a document before asking.");

  renderExamples();
}

function renderExamples() {
  if (!library.documents.length) { els.examples.hidden = true; return; }

  const prompts = [
    "What is this document about?",
    "Summarise the key points",
    "What are the main obligations?",
  ];
  els.examples.hidden = false;
  els.examples.innerHTML = prompts
    .map((p) => `<li class="example">${esc(p)}</li>`).join("");
  els.examples.querySelectorAll(".example").forEach((chip) => {
    chip.onclick = () => { els.q.value = chip.textContent; els.q.focus(); ask(); };
  });
}

async function refresh() {
  try {
    library = await api("/api/status");
    els.modeDot.className = "dot is-ok";
    els.modeText.textContent = library.answers_with_ai
      ? `${library.retriever} · ${library.provider}`
      : `${library.retriever} · extractive`;
    renderLibrary();
  } catch {
    els.modeDot.className = "dot";
    els.modeText.textContent = "Offline";
  }
}

async function removeDoc(docId) {
  try {
    library = await api(`/api/documents/${docId}`, { method: "DELETE" });
    renderLibrary();
  } catch (error) {
    note(error.message, true);
  }
}

/* ─────────────────────────── upload ─────────────────────────── */

async function upload(fileList) {
  const files = [...fileList];
  if (!files.length) return;

  els.drop.classList.add("is-busy");
  note(`Reading ${plural(files.length, "file")}…`);

  const body = new FormData();
  files.forEach((file) => body.append("files", file));

  try {
    library = await api("/api/documents", { method: "POST", body });
    renderLibrary();
    note("");
    els.q.focus();
  } catch (error) {
    note(error.message, true);
  } finally {
    els.drop.classList.remove("is-busy");
    els.fileInput.value = "";
  }
}

els.drop.onclick = () => els.fileInput.click();
els.drop.onkeydown = (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    els.fileInput.click();
  }
};
els.fileInput.onchange = () => upload(els.fileInput.files);

["dragenter", "dragover"].forEach((type) =>
  els.drop.addEventListener(type, (event) => {
    event.preventDefault();
    els.drop.classList.add("is-over");
  }));

["dragleave", "drop"].forEach((type) =>
  els.drop.addEventListener(type, (event) => {
    event.preventDefault();
    els.drop.classList.remove("is-over");
  }));

els.drop.addEventListener("drop", (event) => upload(event.dataTransfer.files));

/* ─────────────────────────── asking ─────────────────────────── */

function addTurn(html, className) {
  els.opening?.remove();
  const node = document.createElement("div");
  node.className = `turn ${className}`;
  node.innerHTML = html;
  els.thread.appendChild(node);
  els.thread.scrollTop = els.thread.scrollHeight;
  return node;
}

/** Replace [1] in the answer body with clickable markers. */
function linkRefs(text) {
  return esc(text).replace(/\[(\d{1,2})\]/g,
    (_, n) => `<span class="ref" data-ref="${n}" role="button" tabindex="0">${n}</span>`);
}

function renderCitations(citations) {
  if (!citations.length) return "";
  return `
    <div class="sources">
      <div class="sources__label">Where this came from</div>
      ${citations.map((c, i) => `
        <div class="slip" data-slip="${c.marker}" style="--i:${i}">
          <div class="slip__head">
            <span class="slip__n">[${c.marker}]</span>
            <span class="slip__file" title="${esc(c.filename)}">${esc(c.filename)}</span>
            <span class="slip__page">page ${c.page}</span>
          </div>
          <p class="slip__quote">${esc(c.quote)}</p>
        </div>`).join("")}
    </div>`;
}

function markCitedDocs(citations) {
  const cited = new Set(citations.map((c) => c.filename));
  els.docs.querySelectorAll(".doc").forEach((li) => {
    li.classList.toggle("is-cited", cited.has(li.dataset.name));
  });
}

async function ask() {
  const question = els.q.value.trim();
  if (!question || busy || !library.documents.length) return;

  busy = true;
  els.send.disabled = true;
  els.q.value = "";
  els.q.style.height = "auto";

  addTurn(`<p>${esc(question)}</p>`, "turn--you");

  // Reading-the-page skeleton rather than bouncing dots: it matches
  // what the system is actually doing and sets the right expectation.
  const pending = addTurn(
    `<div class="reply">
       <div class="scanning">
         <div class="scanning__line"></div>
         <div class="scanning__line"></div>
         <div class="scanning__line"></div>
         <div class="scanning__line"></div>
         <div class="scanning__label">Reading your documents</div>
       </div>
     </div>`,
    "turn--reply"
  );

  try {
    const result = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const flag = result.grounded ? "" :
      `<div class="reply__flag">Not found in your documents</div>`;

    pending.innerHTML = `
      <div class="reply ${result.grounded ? "" : "is-ungrounded"}">
        ${flag}
        <p class="reply__text">${linkRefs(result.answer)}</p>
        ${renderCitations(result.citations)}
      </div>`;

    markCitedDocs(result.citations);

    // Clicking an inline [1] lights up the matching slip.
    pending.querySelectorAll(".ref").forEach((ref) => {
      const highlight = () => {
        const slip = pending.querySelector(`[data-slip="${ref.dataset.ref}"]`);
        if (!slip) return;
        pending.querySelectorAll(".slip").forEach((s) => s.classList.remove("is-lit"));
        // Force a reflow so the flash replays on a repeat click.
        void slip.offsetWidth;
        slip.classList.add("is-lit");
        slip.scrollIntoView({ block: "nearest", behavior: "smooth" });
      };
      ref.onclick = highlight;
      ref.onkeydown = (e) => { if (e.key === "Enter") highlight(); };
    });

    note(`answered in ${result.elapsed_ms} ms · confidence ${result.confidence}`);
  } catch (error) {
    pending.innerHTML =
      `<div class="reply is-ungrounded"><p class="reply__text">${esc(error.message)}</p></div>`;
    note("Something went wrong. Check the server log.", true);
  } finally {
    busy = false;
    els.send.disabled = !library.documents.length;
    els.thread.scrollTop = els.thread.scrollHeight;
  }
}

els.form.onsubmit = (event) => { event.preventDefault(); ask(); };

els.q.addEventListener("input", () => {
  els.q.style.height = "auto";
  els.q.style.height = `${Math.min(els.q.scrollHeight, 160)}px`;
});

els.q.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    ask();
  }
});

refresh();