"use strict";

(() => {
  const REMOTE_BASE = "https://huggingface.co/datasets/ankanmbz/worldguide-results/resolve/main/";
  const $ = (id) => document.getElementById(id);
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  let manifest, results, sampleId, comparisonModel = "MiniMax-H3", benchmark = "worldguide";
  let pair = [], playbackVersion = 0;
  const heroObservers = [];
  const steps = [
    ["CONTEXTPLANNER", "Choose the next atomic action.", "The planner reads the goal and the latest three clip–action pairs, then predicts an interpretable language-level atomic action. Each instruction describes the intended next step and is grounded in visible progress.", "Goal + recent visual and action history", "Next atomic action"],
    ["VIDEO EXECUTOR", "Turn the instruction into a visual action.", "The fine-tuned HunyuanVideo-1.5 Executor generates a short clip from the current visual state, the planner’s action embeddings, and compressed visual memory.", "Action + current visual state + memory", "Generated action clip"],
    ["GENERATED VISUAL FEEDBACK", "Look at what the action produced.", "The generated clip updates the planner’s recent clip–action history and the Executor’s latent memory. Future decisions depend on this observed result, including incomplete execution.", "Generated clip + action", "Updated planner context and Executor memory"],
    ["LEARNED TERMINATION", "Take another step—or call it done.", "The ContextPlanner predicts another action or the completion token from generated progress. The completion token stops generation immediately; an 80-step cap bounds the rollout.", "Updated visual progress + task goal", "Next action or <|Task Completed|>"],
  ];

  function node(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
  }
  function assetURL(path) {
    return REMOTE_BASE + path.split("/").map(encodeURIComponent).join("/");
  }
  function modelLabel(id) {
    return id === "Barnini" ? "Bernini*" : manifest.models.find(model => model.id === id)?.label || id;
  }
  function posterURL(sample, model = "Our") {
    return model === "Our" ? `assets/posters/${sample.id}.webp` : assetURL(sample.videos[model].poster);
  }
  function validManifest(data) {
    return Array.isArray(data?.models) && Array.isArray(data?.samples) && data.samples.length > 0 && data.samples.every(sample =>
      typeof sample.id === "string" && typeof sample.title === "string" && sample.videos?.Our &&
      Object.values(sample.videos).every(video => typeof video.path === "string" && typeof video.poster === "string" && Number.isFinite(video.duration)));
  }
  async function fetchJSON(url, timeout = 7000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(url, { signal: controller.signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } finally { clearTimeout(timer); }
  }

  function setupTabs(selector, callback) {
    const buttons = [...document.querySelectorAll(selector)];
    buttons.forEach(button => {
      button.addEventListener("click", () => {
        buttons.forEach(peer => {
          peer.setAttribute("aria-selected", String(peer === button));
          peer.tabIndex = peer === button ? 0 : -1;
        });
        callback(button);
      });
      button.addEventListener("keydown", event => {
        const current = buttons.indexOf(button);
        let next;
        if (["ArrowRight", "ArrowDown"].includes(event.key)) next = (current + 1) % buttons.length;
        if (["ArrowLeft", "ArrowUp"].includes(event.key)) next = (current - 1 + buttons.length) % buttons.length;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = buttons.length - 1;
        if (next !== undefined) { event.preventDefault(); buttons[next].focus(); buttons[next].click(); }
      });
    });
  }

  setupTabs("[data-step]", button => {
    const [kicker, title, description, input, output] = steps[Number(button.dataset.step)];
    $("loop-kicker").textContent = kicker;
    $("loop-title").textContent = title;
    $("loop-description").textContent = description;
    $("loop-input").textContent = input;
    $("loop-output").textContent = output;
    $("loop-panel").setAttribute("aria-labelledby", button.id);
  });

  const figureDialog = $("figure-dialog");
  document.querySelectorAll("[data-figure]").forEach(button => {
    button.addEventListener("click", () => {
      const source = button.querySelector("img");
      $("expanded-figure").src = source.src;
      $("expanded-figure").alt = source.alt;
      $("expanded-caption").textContent = button.closest("figure").querySelector("figcaption").textContent;
      figureDialog.showModal();
    });
  });
  $("close-figure").addEventListener("click", () => figureDialog.close());
  figureDialog.addEventListener("click", event => {
    if (event.target !== figureDialog) return;
    const rect = figureDialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) figureDialog.close();
  });

  function setupHero() {
    document.querySelectorAll(".reel-media video").forEach(video => {
      const sample = manifest.samples.find(item => item.id === video.dataset.sample);
      if (!sample) return;
      const button = document.querySelector(`[data-video="${video.id}"]`);
      let manuallyPaused = false;
      const load = () => { if (!video.hasAttribute("src")) video.src = assetURL(sample.videos.Our.path); };
      const updateButton = () => {
        button.textContent = video.paused ? "▶" : "Ⅱ";
        button.setAttribute("aria-label", `${video.paused ? "Play" : "Pause"} ${sample.title} video`);
      };
      video.addEventListener("play", updateButton);
      video.addEventListener("pause", updateButton);
      video.addEventListener("error", () => {
        button.textContent = "↻";
        button.setAttribute("aria-label", `Retry ${sample.title} video`);
      });
      button.addEventListener("click", async () => {
        if (!video.paused) { manuallyPaused = true; video.pause(); return; }
        manuallyPaused = false;
        load();
        if (video.error) video.load();
        try { await video.play(); } catch { updateButton(); }
      });
      const observer = new IntersectionObserver(entries => {
        for (const entry of entries) {
          if (!entry.isIntersecting) { video.pause(); continue; }
          if (!reducedMotion.matches && !navigator.connection?.saveData && !manuallyPaused && !document.hidden) {
            load(); video.play().catch(updateButton);
          }
        }
      }, { threshold: .35 });
      observer.observe(video);
      heroObservers.push(observer);
    });
  }

  function makeVideo(sample, model, container) {
    const metadata = sample.videos[model];
    const video = node("video");
    video.controls = true;
    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;
    video.preload = "none";
    video.poster = posterURL(sample, model);
    video.src = assetURL(metadata.path);
    video.setAttribute("aria-label", `${modelLabel(model)}: ${sample.title}`);
    const error = node("div", "video-error");
    error.hidden = true;
    error.setAttribute("role", "status");
    error.append(node("p", "", "This video couldn’t load. Please retry to watch it here."));
    const retry = node("button", "button secondary", "Retry video ↻");
    retry.type = "button";
    retry.addEventListener("click", async () => {
      error.hidden = true;
      video.load();
      try { await video.play(); } catch { error.hidden = false; }
    });
    error.append(retry);
    video.addEventListener("error", () => { error.hidden = false; });
    video.addEventListener("playing", () => { error.hidden = true; });
    ["play", "pause", "ended"].forEach(name => video.addEventListener(name, updatePairButton));
    container.replaceChildren(video, error);
    return video;
  }

  function updatePairButton() {
    const playing = pair.some(video => !video.paused && !video.ended);
    $("play-pair").textContent = playing ? "Pause both Ⅱ" : "Play both ▶";
  }
  function selectSample(id, updateURL = true) {
    const sample = manifest.samples.find(item => item.id === id) || manifest.samples[0];
    sampleId = sample.id;
    playbackVersion++;
    pair.forEach(video => { video.pause(); video.removeAttribute("src"); video.load(); });
    pair = [];
    if (!sample.videos[comparisonModel]) comparisonModel = manifest.models.find(model => model.id !== "Our" && sample.videos[model.id])?.id;
    if (!comparisonModel) { $("playback-status").textContent = "No comparison output is available for this sample."; return; }
    pair = [makeVideo(sample, "Our", $("ours-player")), makeVideo(sample, comparisonModel, $("baseline-player"))];
    $("sample-select").value = sample.id;
    $("model-select").value = comparisonModel;
    [...$("model-select").options].forEach(option => { option.disabled = !sample.videos[option.value]; });
    $("baseline-name").textContent = modelLabel(comparisonModel);
    $("ours-duration").textContent = `${sample.videos.Our.duration.toFixed(1)}s`;
    $("baseline-duration").textContent = `${sample.videos[comparisonModel].duration.toFixed(1)}s`;
    const craft = /^(buildingblock_|paper_airplane_|paper_boat_)/.test(sample.id);
    const ownPlan = ["Barnini", "TempAct", "PhysAgent"].includes(comparisonModel);
    $("baseline-condition").textContent = craft || ownPlan ? "Initial image + task goal" : "Initial image + reference actions";
    $("playback-status").textContent = `${sample.title} · ${sample.category} · Ready to play.`;
    $("protocol-note").textContent = `${craft ? "Video-CraftBench: both models receive the initial image and task goal." : ownPlan ? "WorldGuide Bench: both methods use their own planning process from the initial image and goal." : "WorldGuide Bench: this baseline receives reference actions; WorldGuide predicts its own."} Videos retain their original durations. Paired playback starts both videos; it does not align their procedural steps.`;
    document.querySelectorAll(".sample-choice").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.sample === sample.id)));
    updatePairButton();
    if (updateURL) {
      const url = new URL(location.href);
      url.searchParams.set("sample", sample.id);
      url.searchParams.set("model", comparisonModel);
      history.replaceState(null, "", url);
    }
  }

  async function playPair(restart = false) {
    const version = ++playbackVersion;
    if (restart) pair.forEach(video => { video.currentTime = 0; });
    $("playback-status").textContent = "Loading the selected videos…";
    const outcomes = await Promise.allSettled(pair.map(video => {
      if (video.ended) video.currentTime = 0;
      return video.play();
    }));
    if (version !== playbackVersion) return;
    const failed = outcomes.filter(outcome => outcome.status === "rejected").length;
    $("playback-status").textContent = failed ? "Some videos could not start. Use the individual play or retry controls." : "Both videos playing at their original speed. Different durations and steps are preserved.";
    updatePairButton();
  }
  $("play-pair").addEventListener("click", () => {
    if (pair.some(video => !video.paused && !video.ended)) {
      playbackVersion++;
      pair.forEach(video => video.pause());
      $("playback-status").textContent = "Both videos paused.";
    } else playPair();
  });
  $("restart-pair").addEventListener("click", () => playPair(true));
  $("sample-select").addEventListener("change", event => selectSample(event.target.value));
  $("model-select").addEventListener("change", event => { comparisonModel = event.target.value; selectSample(sampleId); });

  function initializeCollection(data) {
    manifest = data;
    $("sample-total").textContent = manifest.samples.length;
    const params = new URLSearchParams(location.search);
    const requestedModel = params.get("model");
    if (manifest.models.some(model => model.id === requestedModel && model.id !== "Our")) comparisonModel = requestedModel;
    const groups = new Map();
    const fragment = document.createDocumentFragment();
    const duplicateCount = new Map();
    for (const sample of manifest.samples) {
      if (!groups.has(sample.category)) {
        const group = node("optgroup"); group.label = sample.category; groups.set(sample.category, group);
      }
      const occurrence = (duplicateCount.get(sample.title) || 0) + 1;
      duplicateCount.set(sample.title, occurrence);
      const name = sample.title + (occurrence > 1 ? ` · example ${occurrence}` : "");
      const option = node("option", "", name); option.value = sample.id; groups.get(sample.category).append(option);
      const button = node("button", "sample-choice");
      button.type = "button";
      button.dataset.sample = sample.id;
      button.setAttribute("aria-pressed", "false");
      button.setAttribute("aria-label", `Compare ${name}`);
      const img = node("img");
      img.src = posterURL(sample); img.alt = ""; img.loading = "lazy"; img.width = 128; img.height = 78;
      img.addEventListener("error", () => { if (!img.dataset.fallback) { img.dataset.fallback = "true"; img.src = assetURL(sample.videos.Our.poster); } });
      button.append(img, node("span", "", name), node("small", "", sample.category));
      button.addEventListener("click", () => selectSample(sample.id));
      fragment.append(button);
    }
    $("sample-strip").replaceChildren(fragment);
    $("sample-select").replaceChildren(...groups.values());
    $("model-select").replaceChildren(...manifest.models.filter(model => model.id !== "Our").map(model => {
      const option = node("option", "", modelLabel(model.id)); option.value = model.id; return option;
    }));
    ["sample-select", "model-select", "play-pair", "restart-pair"].forEach(id => { $(id).disabled = false; });
    selectSample(params.get("sample") || "fried_rice__4aIJpuEPdOo", false);
    setupHero();
  }

  function renderResults() {
    if (!results) return;
    const craft = benchmark === "craft";
    const rows = results[benchmark];
    $("main-score").replaceChildren(document.createTextNode(craft ? "47.69" : "33.33"), node("span", "", "%"));
    $("score-delta").textContent = craft ? "+14.96%" : "+3.43%";
    $("score-delta").setAttribute("aria-label", (craft ? "14.96" : "3.43") + " percentage points");
    $("score-description").textContent = craft ? "All models start from an initial image and task goal, without reference actions." : "WorldGuide selects its actions from the task goal and generated history.";
    $("benchmark-protocol").textContent = craft ? "Video-CraftBench evaluates 294 block-assembly and paper-folding samples. Every model receives only the initial image and task goal." : "WorldGuide and starred planner–executor baselines receive the initial image and task goal. Unstarred baselines receive reference action plans. 980 test samples.";
    $("benchmark-caveat").textContent = craft ? "" : "The reported WorldGuide–MiniMax-H3 difference on this benchmark is not statistically significant (p = 0.10); the input conditions also differ.";
    $("benchmark-caveat").hidden = craft;
    const selected = craft ? ["WorldGuide", "MiniMax-H3", "TempAct", "HunyuanVideo", "Open-Sora"] : ["WorldGuide", "MiniMax-H3", "HunyuanVideo", "Cosmos", "SpMem"];
    const bars = selected.map(prefix => {
      const row = rows.find(item => item.model.startsWith(prefix));
      const el = node("div", `chart-row${prefix === "WorldGuide" ? " ours" : ""}`);
      const top = node("div", "chart-row-top");
      top.append(node("span", "", row.model), node("strong", "", `${row.values[9].toFixed(2)}%`));
      const track = node("div", "chart-track");
      track.setAttribute("aria-hidden", "true");
      const fill = node("div", "chart-fill");
      fill.style.width = `${row.values[9] / 60 * 100}%`;
      track.append(fill); el.append(top, track); return el;
    });
    const scale = node("div", "chart-scale");
    ["0%", "20%", "40%", "60%"].forEach(label => scale.append(node("span", "", label)));
    $("benchmark-chart").replaceChildren(...bars, scale);

  }
  setupTabs("[data-full-benchmark]", button => {
    const template = $("full-rows-" + button.dataset.fullBenchmark);
    $("results-table").querySelector("tbody").replaceChildren(template.content.cloneNode(true));
    $("full-results-caption").textContent = button.textContent;
    $("full-results-protocol").textContent = template.dataset.protocol;
    $("full-results-panel").setAttribute("aria-labelledby", button.id);
    $("results-table").parentElement.setAttribute("aria-label", button.textContent + "; scroll horizontally");
  });

  setupTabs("[data-benchmark]", button => {
    benchmark = button.dataset.benchmark;
    $("benchmark-panel").setAttribute("aria-labelledby", button.id);
    renderResults();
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      playbackVersion++;
      document.querySelectorAll("video").forEach(video => video.pause());
    }
  });
  reducedMotion.addEventListener("change", event => {
    if (event.matches) document.querySelectorAll(".reel-media video").forEach(video => video.pause());
  });

  async function loadCollection() {
    // Try the hosted manifest first; keep the shipped snapshot available on network failure.
    const localRequest = fetchJSON("manifest.json").then(data => validManifest(data) ? data : null).catch(() => null);
    try {
      const data = await fetchJSON(REMOTE_BASE + "manifest.json", 4500);
      if (!validManifest(data)) throw new Error("Invalid remote manifest");
      initializeCollection(data);
    } catch {
      const local = await localRequest;
      if (local) initializeCollection(local);
      else $("playback-status").textContent = "The video collection couldn’t load. Reload the page or open the full comparison using the link below.";
    }
  }
  fetchJSON("resources.json").then(links => {
    document.querySelectorAll("[data-resource]").forEach(link => {
      const destination = links[link.dataset.resource];
      if (typeof destination !== "string" || !destination.trim()) return;
      let url;
      try { url = new URL(destination, location.href); } catch { return; }
      if (!["http:", "https:"].includes(url.protocol)) return;
      link.href = url.href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.removeAttribute("aria-disabled");
      link.removeAttribute("role");
      link.querySelector(".resource-status").textContent = "↗";
    });
  }).catch(() => { /* Keep the clearly labeled placeholders if no links are configured. */ });
  $("copy-citation").addEventListener("click", async () => {
    const text = $("citation-text").textContent;
    try {
      await navigator.clipboard.writeText(text);
      $("citation-copy-status").textContent = "Citation copied.";
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents($("citation-text"));
      selection.removeAllRanges();
      selection.addRange(range);
      $("citation-copy-status").textContent = "Press Ctrl+C or ⌘C to copy the selected citation.";
    }
  });

  loadCollection();
  fetchJSON("results.json").then(data => { results = data; renderResults(); }).catch(() => {
    $("benchmark-chart").textContent = "The detailed result data couldn’t load. Reload the page to view the chart. Full results remain available below.";
  });

  // Keep the top navigation and side rail in sync with reading position.
  const sectionLinks = [...document.querySelectorAll("[data-section-link]")];
  const readingSections = [...new Set(sectionLinks.map(link => link.dataset.sectionLink))]
    .map(id => document.getElementById(id)).filter(Boolean);
  const pageHeader = document.querySelector(".header");
  let navigationFrame = 0;
  let activeSection = "";
  function updateReadingPosition() {
    navigationFrame = 0;
    let current = readingSections[0];
    const threshold = pageHeader.offsetHeight + 110;
    for (const section of readingSections) {
      if (section.getBoundingClientRect().top <= threshold) current = section;
    }
    if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 4) {
      current = readingSections[readingSections.length - 1];
    }
    pageHeader.classList.toggle("is-scrolled", window.scrollY > 20);
    if (!current || current.id === activeSection) return;
    activeSection = current.id;
    sectionLinks.forEach(link => {
      if (link.dataset.sectionLink === activeSection) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
    const topLink = document.querySelector('.top-nav [aria-current]');
    if (topLink) {
      const nav = topLink.parentElement;
      nav.scrollLeft = topLink.offsetLeft - nav.offsetLeft - (nav.clientWidth - topLink.offsetWidth) / 2;
    }
  }
  function scheduleNavigationUpdate() {
    if (!navigationFrame) navigationFrame = requestAnimationFrame(updateReadingPosition);
  }
  window.addEventListener("scroll", scheduleNavigationUpdate, { passive: true });
  window.addEventListener("resize", scheduleNavigationUpdate);
  window.addEventListener("load", scheduleNavigationUpdate);
  updateReadingPosition();
})();
