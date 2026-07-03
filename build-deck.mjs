import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const OUT = 'decks/journey-of-autorag';
mkdirSync(OUT, { recursive: true });

const HEAD = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
:root{
  --bg:#080c10;--panel:#0d131a;--stroke:#24313d;--stroke2:#2a3a46;
  --ink:#eef4f8;--muted:#8d9aa7;--dim:#5f6b78;
  --green:#65f2ad;--amber:#ffd166;--red:#ff7b72;--cyan:#7dd9ff;
}
body{
  width:720pt;height:405pt;overflow:hidden;position:relative;
  font-family:'Pretendard',sans-serif;color:var(--ink);
  background-color:var(--bg);
  background-image:linear-gradient(#0f1720 1px,transparent 1px),linear-gradient(90deg,#0f1720 1px,transparent 1px);
  background-size:26pt 26pt;
}
.slide{position:absolute;inset:0;padding:30pt 40pt 34pt;display:flex;flex-direction:column;justify-content:center;}
.mono{font-family:'Space Mono',monospace;}
strong{color:var(--green);font-weight:700;}
em{color:var(--cyan);font-style:normal;font-weight:700;}
b{color:var(--ink);font-weight:700;}
h1{font-size:31pt;font-weight:800;line-height:1.2;letter-spacing:-0.5pt;padding-bottom:1pt;}
h2{font-size:19pt;font-weight:800;line-height:1.26;letter-spacing:-0.3pt;padding-bottom:1pt;}
h3{font-size:11pt;font-weight:800;line-height:1.3;}
.kicker{font-family:'Space Mono',monospace;color:var(--green);font-weight:700;font-size:7.5pt;letter-spacing:1.6pt;text-transform:uppercase;margin-bottom:9pt;}
.kicker.amber{color:var(--amber);}.kicker.cyan{color:var(--cyan);}.kicker.red{color:var(--red);}
.sub{color:var(--muted);font-weight:500;font-size:9pt;line-height:1.4;}
.evt{position:absolute;top:20pt;left:40pt;right:40pt;display:flex;justify-content:space-between;font-family:'Space Mono',monospace;font-size:7.5pt;color:var(--muted);}
.footer{position:absolute;bottom:14pt;left:40pt;right:40pt;display:flex;justify-content:space-between;font-family:'Space Mono',monospace;font-size:7pt;color:var(--dim);}
.footer .step{color:var(--green);}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20pt;align-items:center;}
.grid2>*{min-width:0;}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12pt;align-items:stretch;}
.grid3>*{min-width:0;}
.card{background:var(--panel);border:1px solid var(--stroke);border-radius:11pt;padding:13pt 15pt;max-width:100%;}
.tag{display:inline-block;font-family:'Space Mono',monospace;font-size:7pt;font-weight:700;color:var(--green);background:#111923;border:1px solid var(--stroke2);border-radius:14pt;padding:3.5pt 9pt;margin:2.5pt 3pt 2.5pt 0;}
.tag.amber{color:var(--amber);}.tag.cyan{color:var(--cyan);}.tag.red{color:var(--red);}.tag.dim{color:var(--muted);}
.imgwrap{background:#fff;border:1px solid var(--stroke2);border-radius:9pt;padding:6pt;text-align:center;}
.imgwrap.dark{background:#0a0a0a;}
.cap{color:var(--dim);font-size:6.5pt;font-weight:600;margin-top:6pt;font-family:'Space Mono',monospace;}
.term{background:#0a1118;border:1px solid var(--stroke2);border-radius:9pt;overflow:hidden;font-family:'Space Mono',monospace;}
.term .bar{background:#111923;border-bottom:1px solid var(--stroke2);padding:5pt 9pt;display:flex;align-items:center;gap:5pt;}
.term .bar i{width:6pt;height:6pt;border-radius:50%;display:inline-block;}
.term .bar .t{margin-left:6pt;color:var(--muted);font-size:7pt;font-weight:700;}
.term .body{padding:9pt 12pt;font-size:8pt;line-height:1.55;word-break:break-word;}
.term .body .c{color:var(--dim);}.term .body .g{color:var(--green);}.term .body .a{color:var(--amber);}
.term .body .cy{color:var(--cyan);}.term .body .r{color:var(--red);}
.statrow{display:flex;gap:11pt;flex-wrap:wrap;}
.stat{flex:1;min-width:80pt;background:var(--panel);border:1px solid var(--stroke);border-radius:9pt;padding:10pt 12pt;}
.stat .n{font-size:18pt;font-weight:800;color:var(--green);line-height:1;}
.stat .n.cy{color:var(--cyan);}.stat .n.am{color:var(--amber);}.stat .n.ink{color:var(--ink);font-size:13pt;}
.stat .l{color:var(--muted);font-size:6.8pt;font-weight:600;margin-top:5pt;}
.eras{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:7pt;align-items:stretch;}
.era{background:var(--panel);border:1px solid var(--stroke);border-radius:10pt;padding:10pt 11pt;}
.era.active{border-color:var(--green);box-shadow:0 0 0 1px var(--green) inset;}
.era .yr{font-family:'Space Mono',monospace;font-size:6.5pt;color:var(--dim);font-weight:700;}
.era .nm{font-size:11pt;font-weight:800;margin:4pt 0;color:var(--ink);}
.era .ds{color:var(--muted);font-size:7pt;line-height:1.32;font-weight:600;}
.era .goal{margin-top:7pt;font-family:'Space Mono',monospace;font-size:6.3pt;color:var(--green);font-weight:700;}
.arw{display:flex;align-items:center;justify-content:center;color:var(--green);font-size:15pt;font-weight:800;}
.bigq{font-size:14pt;font-weight:800;line-height:1.28;color:var(--ink);}
.bigq .hl{color:var(--green);}
.list{list-style:none;}
.list li{font-size:8.5pt;color:var(--muted);font-weight:600;line-height:1.32;margin:4pt 0;padding-left:12pt;position:relative;}
.list li:before{content:"";position:absolute;left:0;top:5pt;width:4pt;height:4pt;border-radius:50%;background:var(--green);}
/* architecture */
.stack{display:flex;flex-direction:column;gap:5pt;}
.lyr{border-radius:8pt;padding:7pt 11pt;display:flex;align-items:center;gap:9pt;border:1px solid var(--stroke2);background:var(--panel);}
.lyr .id{font-family:'Space Mono',monospace;font-weight:800;font-size:8pt;color:var(--dim);min-width:20pt;}
.lyr .nm{font-weight:800;font-size:9pt;color:var(--ink);}
.lyr .nm small{display:block;color:var(--muted);font-size:6.8pt;font-weight:600;margin-top:1pt;}
.lyr.l3{border-color:#2b4a5c;}.lyr.l4{border-color:#2f5f86;}.lyr.top{border-color:#2e6b4d;}
.endcap{text-align:center;font-family:'Space Mono',monospace;font-size:6.8pt;color:var(--dim);font-weight:700;padding:4pt;}
.adesc .d{display:flex;gap:8pt;margin:5pt 0;align-items:flex-start;}
.adesc .d .k{font-family:'Space Mono',monospace;font-weight:800;font-size:7pt;color:var(--green);min-width:26pt;padding-top:1.5pt;}
.adesc .d .v{font-size:7.6pt;line-height:1.3;color:var(--muted);font-weight:600;}
.adesc .d .v b{display:block;}
/* module blocks */
.mods{display:grid;grid-template-columns:1fr auto 1.2fr auto 1fr;gap:8pt;align-items:center;}
.mods .plus{color:var(--green);font-size:14pt;font-weight:800;text-align:center;}
.modbox{background:var(--panel);border:1px solid var(--stroke);border-radius:9pt;padding:9pt 10pt;}
.modbox.main{border-color:var(--cyan);box-shadow:0 0 0 1px var(--cyan) inset;text-align:center;}
.modbox .cat{font-family:'Space Mono',monospace;font-size:6pt;font-weight:700;color:var(--green);text-transform:uppercase;letter-spacing:.6pt;}
.modbox .ttl{font-size:10pt;font-weight:800;margin:3pt 0 5pt;color:var(--ink);}
.modbox .li{font-size:6.6pt;color:var(--muted);font-weight:600;line-height:1.3;}
.pipe{display:flex;gap:3pt;justify-content:center;margin-top:6pt;flex-wrap:wrap;}
.pipe span{font-family:'Space Mono',monospace;font-size:6pt;background:#111923;border:1px solid var(--stroke2);border-radius:5pt;padding:3pt 5pt;color:var(--cyan);font-weight:700;}
/* bars */
.bars{display:flex;flex-direction:column;gap:9pt;}
.barrow{display:grid;grid-template-columns:52pt 1fr;gap:9pt;align-items:center;}
.barrow .lab{font-family:'Space Mono',monospace;font-size:7pt;color:var(--muted);font-weight:700;text-align:right;}
.bartrack{display:flex;flex-direction:column;gap:4pt;}
.barline{display:flex;align-items:center;gap:6pt;}
.bar{height:13pt;border-radius:4pt;flex-shrink:0;}
.bar.raw{background:linear-gradient(90deg,#ff7b72,#b3554f);}
.bar.jk{background:linear-gradient(90deg,#65f2ad,#2fae78);}
.bval{font-family:'Space Mono',monospace;font-size:7pt;font-weight:800;white-space:nowrap;}
.bval.raw{color:var(--red);}.bval.jk{color:var(--green);}
.legend{display:flex;gap:14pt;margin-top:3pt;font-family:'Space Mono',monospace;font-size:7pt;font-weight:700;}
.legend .raw{color:var(--red);}.legend .jk{color:var(--green);}
.tcenter{text-align:center;align-items:center;}
</style>
</head>
<body>`;

const FOOT = `</body>
</html>`;

function slide(inner) { return HEAD + '\n<div class="slide">\n' + inner + '\n</div>\n' + FOOT; }
function footer(label, step) { return `<div class="footer"><span>${label}</span><span class="step">${step}</span></div>`; }

const slides = [];

// 01 TITLE
slides.push(slide(`
<div class="evt"><span>ICML Night · Instruct.KR × Exa — AI Search Technology Meetup, Seoul</span><span>20 min</span></div>
<div class="kicker">The Journey of AutoRAG</div>
<h1>From <em>AutoML for RAG</em><br>to <span style="color:var(--green)">Search Infra for Agents</span></h1>
<p class="sub" style="margin-top:13pt;max-width:520pt;">How a RAG auto-optimization tool with 4.8K stars became a mission to build the<br>knowledge-base layer &amp; the all-purpose librarian search agent that AI agents are missing.</p>
<div style="margin-top:20pt;">
  <span class="tag">Jeffrey Kim · Creator of AutoRAG</span>
  <span class="tag cyan">NomaDamas — AI Hacker House, Seoul</span>
  <span class="tag dim">Marker Inc.</span>
</div>
${footer('Journey of AutoRAG','01')}`));

// 02 WHOAMI
slides.push(slide(`
<div class="kicker">whoami</div>
<div class="grid2" style="gap:26pt;">
  <div>
    <h2>Hi, I build<br>retrieval systems.</h2>
    <p class="sub" style="margin-top:9pt;">I created <strong>AutoRAG</strong> and work out of <em>NomaDamas</em>, an AI hacker house in Seoul. I spend my time on one question:</p>
    <p class="bigq" style="margin-top:12pt;font-size:12pt;">"Which retrieval pipeline is <span class="hl">actually</span> best — for <span class="hl">this</span> data, <span class="hl">this</span> use case?"</p>
  </div>
  <div>
    <div class="term">
      <div class="bar"><i style="background:#ff7b72"></i><i style="background:#ffd166"></i><i style="background:#65f2ad"></i><span class="t">~/nomadamas</span></div>
      <div class="body">
        <div><span class="c">#</span> open-source I ship &amp; maintain</div>
        <div><span class="g">AutoRAG</span> <span class="a">★ 4.8k</span> RAG AutoML</div>
        <div><span class="g">AutoRAG-Research</span> <span class="a">★ 144</span> RAG bench</div>
        <div><span class="g">MinSync</span> <span class="a">★ 51</span> incremental index</div>
        <div><span class="g">agentdir</span> <span class="a">★ 51</span> virtual file tree</div>
        <div><span class="g">jikji</span> <span class="a">★ 35</span> agent file maps</div>
        <div style="margin-top:5pt"><span class="c">#</span> awards</div>
        <div><span class="cy">Minister of Science &amp; ICT Award</span></div>
      </div>
    </div>
  </div>
</div>
${footer('whoami','02')}`));

// 03 JOURNEY
slides.push(slide(`
<div class="kicker">The map for the next 20 minutes</div>
<h2 style="margin-bottom:12pt;font-size:16pt;">One product, three eras — the goal kept changing.</h2>
<div class="eras">
  <div class="era"><div class="yr">2024 →</div><div class="nm">AutoRAG</div><div class="ds">Find the optimal RAG pipeline for <b>your data</b>, automatically.</div><div class="goal">GOAL · AutoML for RAG</div></div>
  <div class="arw">→</div>
  <div class="era"><div class="yr">2025 →</div><div class="nm">AutoRAG-Research</div><div class="ds">Unify datasets, SOTA pipelines &amp; metrics. Make RAG research <b>reproducible</b>.</div><div class="goal">GOAL · Research automation</div></div>
  <div class="arw">→</div>
  <div class="era active"><div class="yr">2026 →</div><div class="nm">AutoRAG 2.0</div><div class="ds">The <b>knowledge-base layer</b> &amp; librarian search agent for the agent era.</div><div class="goal">GOAL · Search infra for agents</div></div>
</div>
<div class="imgwrap" style="margin-top:12pt;">
  <img src="./assets/star-history-render.png" alt="GitHub star history chart" style="max-height:150pt;max-width:100%;width:auto;border-radius:5pt;">
</div>
<div class="cap">GitHub star history · star-history.com — AutoRAG (red) to ~4.8K; the new agent-era repos rising bottom-right.</div>
${footer('The Journey','03')}`));

// 04 WHY
slides.push(slide(`
<div class="kicker amber">2024 · How it started</div>
<div class="grid2" style="gap:26pt;">
  <div>
    <h2>There were already<br>too many frameworks.</h2>
    <p class="sub" style="margin-top:9pt;">RAG frameworks? Dozens. Evaluation frameworks? Plenty. Yet everyone kept asking the same unanswered question:</p>
    <div class="card" style="margin-top:12pt;border-color:var(--amber);">
      <p style="font-size:9pt;font-weight:700;line-height:1.35;">"There are many RAG pipelines &amp; modules out there — but you <span style="color:var(--amber)">don't know which one is best for your own data and use case</span>. Building and evaluating all of them is painfully slow."</p>
    </div>
  </div>
  <div>
    <p class="sub" style="margin-bottom:9pt;">The missing idea wasn't another pipeline. It was the concept from a neighboring field:</p>
    <div class="card tcenter" style="border-color:var(--green);">
      <div class="mono" style="color:var(--dim);font-size:7pt;font-weight:700;">Hyperparameter tuning for ML</div>
      <div style="font-size:15pt;font-weight:800;margin:6pt 0;color:var(--green)">AutoML</div>
      <div class="mono" style="color:var(--green);font-size:8pt;">▼ applied to retrieval ▼</div>
      <div style="font-size:15pt;font-weight:800;margin:6pt 0;color:var(--cyan)">AutoRAG</div>
      <div class="sub" style="font-size:7.5pt;">Search the space of parsers · chunkers · retrievers · rerankers · prompts · LLMs — let evaluation pick the winner.</div>
    </div>
  </div>
</div>
${footer('Why AutoRAG','04')}`));

// 05 AUTOML FOR RAG
slides.push(slide(`
<div class="kicker">AutoRAG · AutoML for RAG</div>
<div class="imgwrap dark" style="margin-bottom:10pt;"><img src="./assets/autorag-thumbnail.png" alt="AutoRAG banner" style="max-height:120pt;max-width:100%;width:auto;"></div>
<div class="statrow">
  <div class="stat"><div class="n am">★ 4.8K</div><div class="l">GitHub stars · 400+ forks</div></div>
  <div class="stat"><div class="n cy" style="font-size:13pt;">Trendshift</div><div class="l">Weekly-trending repository</div></div>
  <div class="stat"><div class="n ink">arXiv</div><div class="l">2410.20878 · published</div></div>
  <div class="stat"><div class="n ink" style="font-size:11pt;">Uber · SKT<br>AWS · Hanwha</div><div class="l">Real-world adopters</div></div>
</div>
<p class="sub" style="margin-top:11pt;">Give AutoRAG your QA + corpus data → it evaluates many RAG module combinations automatically → returns the best pipeline for <em>your</em> use case, ready to deploy.</p>
${footer('AutoRAG','05')}`));

// 06 DATA CREATION
slides.push(slide(`
<div class="kicker">AutoRAG · Step 0 — you need data to optimize on</div>
<h2 style="font-size:15pt;margin-bottom:5pt;">Optimization needs a QA set + a corpus.</h2>
<p class="sub" style="margin-bottom:10pt;">So AutoRAG ships a full data-creation pipeline: <em>parse → chunk → QA generation</em> — turn raw PDFs/docs into an evaluation dataset.</p>
<div class="imgwrap"><img src="./assets/autorag-data-creation.png" alt="AutoRAG data creation flow" style="max-height:130pt;max-width:100%;width:auto;"></div>
<div style="margin-top:10pt;">
  <span class="tag">parse · pdfminer / llama / clova</span>
  <span class="tag">chunk · token / semantic</span>
  <span class="tag">QA · factoid · evolve · filter</span>
  <span class="tag cyan">→ qa.parquet + corpus.parquet</span>
</div>
${footer('AutoRAG · Data','06')}`));

// 07 NODES
slides.push(slide(`
<div class="kicker">AutoRAG · How it optimizes</div>
<h2 style="font-size:15pt;margin-bottom:5pt;">A RAG pipeline = a chain of <em>nodes</em>.</h2>
<p class="sub" style="margin-bottom:10pt;">AutoRAG models the pipeline as ordered node lines. Each node has swappable modules; the framework greedily searches for the best module + params at every stage.</p>
<div class="imgwrap dark"><img src="./assets/autorag-node-structure.png" alt="AutoRAG RAG node structure" style="max-height:130pt;max-width:100%;width:auto;"></div>
<p class="cap">Query → Pre-Retrieval → Retrieval → Post-Retrieval → Prompt → LLM → Answer. Every box is a decision point AutoRAG optimizes.</p>
${footer('AutoRAG · Nodes','07')}`));

// 08 MODULES
slides.push(slide(`
<div class="kicker">AutoRAG · The search space</div>
<h2 style="font-size:14pt;margin-bottom:8pt;">Every node has many modules — AutoRAG tries them for you.</h2>
<div class="imgwrap dark"><img src="./assets/autorag-all-nodes-modules.png" alt="AutoRAG nodes and modules" style="max-height:175pt;max-width:100%;width:auto;"></div>
<p class="sub" style="margin-top:9pt;">Configure the space in one YAML. AutoRAG runs the trials, evaluates with retrieval + generation metrics (F1, Recall, nDCG, MRR, METEOR, ROUGE, SemScore…), and emits a <em>summary.csv</em> with the winning pipeline.</p>
${footer('AutoRAG · Modules','08')}`));

// 09 DEPLOY
slides.push(slide(`
<div class="kicker">AutoRAG · From trial to production</div>
<div class="grid2" style="gap:22pt;">
  <div>
    <h2 style="font-size:15pt;">Optimize → inspect →<br>deploy in one flow.</h2>
    <ul class="list" style="margin-top:9pt;">
      <li>Interactive <em>dashboard</em> to compare every trial &amp; module</li>
      <li>Deploy the winner as <em>code</em>, an <em>API server</em>, or a <em>web app</em></li>
      <li>YAML in, best-pipeline artifact out — fully reproducible</li>
    </ul>
    <div style="margin-top:10pt;">
      <span class="tag">autorag evaluate</span><span class="tag">autorag dashboard</span><span class="tag cyan">autorag run_api</span>
    </div>
  </div>
  <div>
    <div class="imgwrap"><img src="./assets/autorag-dashboard.gif" alt="AutoRAG dashboard" style="max-height:190pt;max-width:100%;width:auto;border-radius:4pt;"></div>
    <div class="cap">AutoRAG optimization dashboard</div>
  </div>
</div>
${footer('AutoRAG · Deploy','09')}`));

// 10 CEILING
slides.push(slide(`
<div class="kicker red">2025 · Where AutoRAG hit its ceiling</div>
<h2 style="margin-bottom:11pt;font-size:17pt;">Then the ground shifted under RAG.</h2>
<div class="grid3">
  <div class="card" style="border-color:#3a2b2b;"><div class="mono" style="color:var(--red);font-size:6.5pt;font-weight:800;">CONSTRAINT 01</div><h3 style="margin:5pt 0;">Locked to "advanced RAG"</h3><p class="sub" style="font-size:7.5pt;">Agentic RAG arrived. AutoRAG's fixed node structure couldn't express the wild variety of new pipelines.</p></div>
  <div class="card" style="border-color:#3a2b2b;"><div class="mono" style="color:var(--red);font-size:6.5pt;font-weight:800;">CONSTRAINT 02</div><h3 style="margin:5pt 0;">Datasets are hard</h3><p class="sub" style="font-size:7.5pt;">Every benchmark has a different format. Building &amp; embedding evaluation data is slow, repetitive, error-prone.</p></div>
  <div class="card" style="border-color:#3a2b2b;"><div class="mono" style="color:var(--red);font-size:6.5pt;font-weight:800;">CONSTRAINT 03</div><h3 style="margin:5pt 0;">Research won't reproduce</h3><p class="sub" style="font-size:7.5pt;">Every paper claims SOTA. Re-implementing each one to compare fairly is a research project by itself.</p></div>
</div>
<p class="bigq" style="margin-top:14pt;font-size:12pt;">A tool that optimizes <span class="hl">one</span> pipeline shape can't answer <span class="hl">"which pipeline shape is even right?"</span></p>
${footer('The Ceiling','10')}`));

// 11 RESEARCH
slides.push(slide(`
<div class="kicker">AutoRAG-Research · Automate your RAG research</div>
<div class="imgwrap" style="background:#6b5cff;padding:0;overflow:hidden;"><img src="./assets/autorag-research-thumbnail.png" alt="AutoRAG-Research banner" style="max-height:190pt;max-width:100%;width:auto;display:block;margin:0 auto;"></div>
<p class="sub" style="margin-top:10pt;">One framework that unifies <em>datasets</em>, <em>SOTA pipelines</em>, and <em>metrics</em> — so you can benchmark your idea against the real state of the art with one command.</p>
<div style="margin-top:4pt;">
  <span class="tag dim">"Every dataset differs" → unified &amp; pre-embedded</span>
  <span class="tag dim">"Every paper claims SOTA" → run them all, compare</span>
</div>
${footer('AutoRAG-Research','11')}`));

// 12 RESEARCH WHAT
slides.push(slide(`
<div class="kicker">AutoRAG-Research · One PostgreSQL, everything pre-built</div>
<div class="grid3" style="gap:11pt;">
  <div class="card"><div class="mono" style="color:var(--green);font-size:6pt;font-weight:800;">DATASETS</div><h3 style="font-size:10pt;margin:4pt 0;">Unified &amp; pre-embedded</h3><p class="sub" style="font-size:7pt;">BEIR · MTEB · RAGBench · MrTyDi · BRIGHT · ViDoRe v1–v3 · VisRAG · Open-RAGBench. Text <em>and</em> image.</p></div>
  <div class="card"><div class="mono" style="color:var(--cyan);font-size:6pt;font-weight:800;">PIPELINES</div><h3 style="font-size:10pt;margin:4pt 0;">SOTA from papers</h3><p class="sub" style="font-size:7pt;">DPR · BM25 · HyDE · Query Rewrite · Hybrid RRF/CC · BasicRAG · IRCoT · ET2RAG · VisRAG · MAIN-RAG.</p></div>
  <div class="card"><div class="mono" style="color:var(--amber);font-size:6pt;font-weight:800;">METRICS</div><h3 style="font-size:10pt;margin:4pt 0;">Retrieval + generation</h3><p class="sub" style="font-size:7pt;">Recall · Precision · F1 · nDCG · MRR · MAP · BLEU · METEOR · ROUGE · BERTScore · SemScore.</p></div>
</div>
<div class="term" style="margin-top:12pt;">
  <div class="bar"><i style="background:#ff7b72"></i><i style="background:#ffd166"></i><i style="background:#65f2ad"></i><span class="t">benchmark in 3 commands</span></div>
  <div class="body" style="font-size:7.5pt;">
    <div><span class="g">$</span> autorag-research ingest <span class="cy">--name beir --extra dataset-name=scifact</span></div>
    <div><span class="g">$</span> autorag-research data restore <span class="cy">beir beir_arguana_test_qwen</span> <span class="c"># pre-computed embeddings</span></div>
    <div><span class="g">$</span> autorag-research run <span class="cy">--db-name=beir_scifact_test</span> <span class="c"># all pipelines, one leaderboard</span></div>
  </div>
</div>
${footer('AutoRAG-Research','12')}`));

// 13 RESEARCH SKILLS
slides.push(slide(`
<div class="kicker cyan">AutoRAG-Research · Built for fast iteration</div>
<div class="grid2" style="gap:24pt;">
  <div>
    <h2 style="font-size:14pt;">Plugins + agent skills =<br>experiment at agent speed.</h2>
    <ul class="list" style="margin-top:9pt;">
      <li><em>Plugin system</em> — scaffold a plugin, <span class="mono">pip install -e .</span>, run beside the built-ins. No fork.</li>
      <li><em>Agent skill</em> — your coding agent queries results straight from PostgreSQL in natural language.</li>
      <li>Spin up dozens of experiments &amp; ask questions conversationally.</li>
    </ul>
    <div style="margin-top:9pt;"><span class="tag">npx skills add NomaDamas/AutoRAG-Research</span></div>
  </div>
  <div>
    <div class="term">
      <div class="bar"><i style="background:#ff7b72"></i><i style="background:#ffd166"></i><i style="background:#65f2ad"></i><span class="t">autorag-query · agent skill</span></div>
      <div class="body" style="font-size:8.5pt;">
        <div><span class="a">You ▸</span> Which pipeline has the best BLEU score?</div>
        <div style="margin-top:7pt;"><span class="g">Agent ▸</span> <span class="cy">hybrid_search_v2</span> achieved the</div>
        <div>highest BLEU score of <span class="g">0.85</span>.</div>
        <div style="margin-top:9pt;" class="c"># queried directly from your results DB</div>
      </div>
    </div>
  </div>
</div>
${footer('AutoRAG-Research','13')}`));

// 14 AGENT ERA
slides.push(slide(`
<div class="slide tcenter" style="position:static;padding:0;">
  <div class="kicker" style="align-self:center;">2026 · and then…</div>
  <h1 style="font-size:29pt;text-align:center;">The agent era arrived —<br>for real this time.</h1>
  <p class="sub" style="max-width:470pt;margin:14pt auto 0;text-align:center;font-size:9pt;">Claude Code. Codex. Autonomous coding &amp; knowledge agents everywhere.<br>So we asked ourselves the only question that mattered:</p>
  <p class="bigq" style="margin-top:16pt;font-size:16pt;text-align:center;">In <span class="hl">this</span> era, what should AutoRAG become?</p>
</div>
${footer('The Agent Era','14')}`));

// 15 VISION + PARTNERS
slides.push(slide(`
<div class="kicker">AutoRAG 2.0 · The answer</div>
<div class="grid2" style="gap:24pt;">
  <div>
    <h2 style="font-size:15pt;">Build the missing<br><em>knowledge-base layer</em> for agents.</h2>
    <p class="sub" style="margin-top:9pt;font-size:8pt;">Agents are brilliant at reasoning and tool use — but the layer that <b>organically integrates &amp; manages RAG and many data sources</b>, and retrieves information from <b>everywhere</b> (including your local files), is still missing.</p>
    <p class="sub" style="margin-top:7pt;font-size:8pt;">AutoRAG 2.0 fills that gap — built with <em>NIPA</em> open-source program support.</p>
  </div>
  <div>
    <div class="card">
      <div class="mono" style="color:var(--dim);font-size:6.5pt;font-weight:800;">CONSORTIUM</div>
      <div style="display:flex;align-items:baseline;gap:7pt;margin:7pt 0 2pt;"><span class="tag">Lead</span><b style="font-size:9pt;">Marker Inc. × NomaDamas</b></div>
      <div class="sub" style="font-size:6.8pt;margin-bottom:6pt;">Core architecture · module development · OSS governance</div>
      <div style="display:flex;align-items:baseline;gap:7pt;margin:2pt 0;"><span class="tag cyan">Partner</span><b style="font-size:9pt;">BrainCrew — TeddyNote</b></div>
      <div class="sub" style="font-size:6.8pt;margin-bottom:6pt;">LangChain Ambassador · agent validation · education</div>
      <div style="display:flex;align-items:baseline;gap:7pt;margin:2pt 0;"><span class="tag amber">Partner</span><b style="font-size:9pt;">2e Consulting</b></div>
      <div class="sub" style="font-size:6.8pt;">Public &amp; finance domain requirements · pilot validation</div>
    </div>
    <div class="cap">Supported by NIPA — 2026 Open-Source AI·SW program</div>
  </div>
</div>
${footer('AutoRAG 2.0 · Vision','15')}`));

// 16 ANALOGY
slides.push(slide(`
<div class="kicker">Why a knowledge-base layer?</div>
<h2 style="font-size:15pt;margin-bottom:11pt;">Right now, org knowledge is flyers on a fridge.</h2>
<div class="grid2" style="gap:18pt;align-items:stretch;">
  <div class="card" style="border-color:#3a2b2b;">
    <div class="mono" style="color:var(--red);font-size:6.5pt;font-weight:800;">TODAY</div>
    <h3 style="font-size:11pt;margin:5pt 0;color:var(--red)">Flyers on the fridge</h3>
    <ul class="list" style="margin-top:3pt;"><li style="font-size:7.6pt;">Dig through a pile to find the menu you want</li><li style="font-size:7.6pt;">Call each restaurant to check what's available</li><li style="font-size:7.6pt;">Closed shops still have flyers up — stale &amp; noisy</li></ul>
  </div>
  <div class="card" style="border-color:#2e6b4d;">
    <div class="mono" style="color:var(--green);font-size:6.5pt;font-weight:800;">WITH AUTORAG 2.0</div>
    <h3 style="font-size:11pt;margin:5pt 0;color:var(--green)">A delivery app</h3>
    <ul class="list" style="margin-top:3pt;"><li style="font-size:7.6pt;">Search the menu you want, instantly</li><li style="font-size:7.6pt;">Prices, options, reviews — categorized</li><li style="font-size:7.6pt;">Closed shops auto-removed — always current</li></ul>
  </div>
</div>
<p class="bigq tcenter" style="margin-top:14pt;font-size:11pt;max-width:600pt;margin-left:auto;margin-right:auto;">An <span class="hl">agent-native knowledge-base layer</span> — the middle layer that lets an agent use your documents at 200%.</p>
${footer('Missing Layer','16')}`));

// 17 ARCHITECTURE
slides.push(slide(`
<div class="kicker">AutoRAG 2.0 · Architecture</div>
<h2 style="font-size:13pt;margin-bottom:9pt;">A 5-layer knowledge-base layer between your files and the agent.</h2>
<div class="grid2" style="gap:22pt;align-items:center;">
  <div>
    <div class="endcap">▲ Upper agent — Claude Code · Codex · OpenClaw ▲</div>
    <div class="stack">
      <div class="lyr top"><span class="id">05</span><span class="nm">Operations <small>self-re-evaluating optimization loop</small></span></div>
      <div class="lyr l4"><span class="id">04</span><span class="nm">Agent Interface <small>CLI &amp; Skills — agents call local knowledge</small></span></div>
      <div class="lyr l3"><span class="id">03</span><span class="nm">Search Orchestration <small>hybrid semantic + keyword + metadata</small></span></div>
      <div class="lyr"><span class="id">02</span><span class="nm">Storage <small>vector DB + virtual FS · incremental indexing</small></span></div>
      <div class="lyr"><span class="id">01</span><span class="nm">Document Understanding <small>open-source parsers · PDF · HWP · DOCX</small></span></div>
    </div>
    <div class="endcap">▼ Local document folder ▼</div>
  </div>
  <div class="adesc">
    <div class="d"><span class="k">L01</span><span class="v"><b>Unstructured parsing</b>Parse tables, images &amp; formulas into logical structure.</span></div>
    <div class="d"><span class="k">L02</span><span class="v"><b>Integrity &amp; freshness</b>Detect file changes in real time; re-index only the delta.</span></div>
    <div class="d"><span class="k">L03</span><span class="v"><b>Intelligent hybrid search</b>Combine methods dynamically to find the best evidence.</span></div>
    <div class="d"><span class="k">L04</span><span class="v"><b>Instant agent connection</b>No bespoke integration — CLI + Skills, ready to call.</span></div>
    <div class="d"><span class="k">L05</span><span class="v"><b>Sustained advantage</b>Learn from usage logs; keep tuning the strategy.</span></div>
    <div class="sub" style="font-size:6.8pt;margin-top:5pt;color:var(--dim)">Each layer is a swappable module — not locked to any vector DB, engine, or model.</div>
  </div>
</div>
${footer('AutoRAG 2.0 · Architecture','17')}`));

// 18 PERSONALIZATION
slides.push(slide(`
<div class="kicker cyan">The evolution of "Auto"</div>
<h2 style="font-size:14pt;margin-bottom:12pt;">Naive optimization → memory-based personalization.</h2>
<div class="grid2" style="gap:20pt;align-items:stretch;">
  <div class="card"><div class="mono" style="color:var(--dim);font-size:6.5pt;font-weight:800;">AutoRAG 1.0</div><h3 style="font-size:11pt;margin:5pt 0;">One-shot search over a config space</h3><p class="sub" style="font-size:8pt;">You bring a static eval set. AutoRAG finds one "best" pipeline. Great — until the data, users, and questions drift.</p></div>
  <div class="card" style="border-color:var(--cyan);"><div class="mono" style="color:var(--cyan);font-size:6.5pt;font-weight:800;">AutoRAG 2.0</div><h3 style="font-size:11pt;margin:5pt 0;">A memory-driven, personalizing loop</h3><p class="sub" style="font-size:8pt;">Operation logs → the system re-evaluates workflows, adapts retrieval to <em>your</em> corpus, queries, feedback. Optimization becomes <b>personalization</b>.</p></div>
</div>
<div class="tcenter" style="margin-top:14pt;display:flex;justify-content:center;align-items:center;gap:8pt;">
  <span class="tag dim">grid search</span><span style="color:var(--green);font-size:13pt;">→</span><span class="tag cyan">memory + feedback + self-tuning</span>
</div>
${footer('Auto → Personal','18')}`));

// 19 BUILDING BLOCKS
slides.push(slide(`
<div class="kicker">The building blocks · being integrated into AutoRAG</div>
<div class="imgwrap" style="margin-bottom:10pt;"><img src="./assets/minsync-flow.svg" alt="MinSync incremental indexing flow" style="max-height:150pt;max-width:100%;width:auto;"></div>
<div class="grid2" style="gap:18pt;align-items:stretch;">
  <div class="card"><div class="mono" style="color:var(--green);font-size:6.5pt;font-weight:800;">MinSync · Rust · ★51</div><h3 style="font-size:10pt;margin:4pt 0;">Incremental indexing</h3><p class="sub" style="font-size:7.5pt;">Git-free change detection via mtime/size/hash. Re-embed only changed chunks, sweep stale ones — the Storage layer stays fresh at near-zero cost.</p></div>
  <div class="card"><div class="mono" style="color:var(--cyan);font-size:6.5pt;font-weight:800;">agentdir · Rust · ★51</div><h3 style="font-size:10pt;margin:4pt 0;">Virtual directories</h3><p class="sub" style="font-size:7.5pt;">Present the same originals in a purpose-built, read-only layout. CoW/reflink — no data copy. Give agents a better working tree without moving files.</p></div>
</div>
${footer('MinSync + agentdir','19')}`));

// 20 JIKJI
slides.push(slide(`
<div class="kicker">jikji · a simple file map, a huge win</div>
<h2 style="font-size:13pt;margin-bottom:4pt;">Prepare a local map first → agents search less, hit more.</h2>
<p class="sub" style="margin-bottom:10pt;font-size:7.5pt;">HippoCamp benchmark · 551 local file-search cases · raw Hermes agent vs. the same agent with <em>jikji find</em>.</p>
<div class="grid2" style="gap:22pt;align-items:center;">
  <div class="bars">
    <div class="barrow"><span class="lab">Hit@1</span><div class="bartrack"><div class="barline"><div class="bar raw" style="width:64%;"></div><span class="bval raw">0.6697 raw</span></div><div class="barline"><div class="bar jk" style="width:76%;"></div><span class="bval jk">0.7949 jikji</span></div></div></div>
    <div class="barrow"><span class="lab">LLM calls</span><div class="bartrack"><div class="barline"><div class="bar raw" style="width:76%;"></div><span class="bval raw">6,420 raw</span></div><div class="barline"><div class="bar jk" style="width:8%;"></div><span class="bval jk">551 jikji</span></div></div></div>
    <div class="barrow"><span class="lab">Tokens</span><div class="bartrack"><div class="barline"><div class="bar raw" style="width:76%;"></div><span class="bval raw">21.3M raw</span></div><div class="barline"><div class="bar jk" style="width:4%;"></div><span class="bval jk">0.25M jikji</span></div></div></div>
    <div class="barrow"><span class="lab">Wall time</span><div class="bartrack"><div class="barline"><div class="bar raw" style="width:76%;"></div><span class="bval raw">31,232s raw</span></div><div class="barline"><div class="bar jk" style="width:5%;"></div><span class="bval jk">1,164s jikji</span></div></div></div>
    <div class="legend"><span class="raw">■ raw Hermes</span><span class="jk">■ + jikji find</span></div>
  </div>
  <div>
    <div class="statrow" style="flex-direction:column;gap:8pt;">
      <div class="stat"><div class="n">86×</div><div class="l">fewer total tokens</div></div>
      <div class="stat"><div class="n cy">11.7×</div><div class="l">fewer LLM calls</div></div>
      <div class="stat"><div class="n am">+12.5pt</div><div class="l">higher Hit@1 accuracy</div></div>
    </div>
    <p class="sub" style="font-size:6.8pt;margin-top:8pt;">Metadata + file maps + parser caches + graph routes → one ranked candidate slate. The agent judges once instead of crawling.</p>
  </div>
</div>
${footer('jikji · file maps','20')}`));

// 21 CLOSING
slides.push(slide(`
<div class="slide tcenter" style="position:static;padding:0;">
  <div class="kicker" style="align-self:center;">Where the journey is heading</div>
  <h1 style="font-size:26pt;text-align:center;">The all-purpose<br><span style="color:var(--green)">librarian search agent.</span></h1>
  <p class="sub" style="max-width:490pt;margin:14pt auto 0;text-align:center;font-size:8.5pt;">One agent that <em>manages many data sources</em>, <em>digs everywhere</em> for information — local files included — and <em>keeps itself current</em> through memory-based personalization.</p>
  <div style="margin-top:16pt;display:flex;justify-content:center;align-items:center;gap:6pt;flex-wrap:wrap;">
    <span class="tag" style="margin:0;">AutoML for RAG</span><span style="color:var(--green)">→</span><span class="tag cyan" style="margin:0;">Research automation</span><span style="color:var(--green)">→</span><span class="tag amber" style="margin:0;">Search infra for agents</span>
  </div>
  <div style="margin-top:12pt;display:flex;justify-content:center;align-items:center;gap:6pt;flex-wrap:wrap;">
    <span class="tag dim" style="margin:0;">github.com/Marker-Inc-Korea/AutoRAG</span><span class="tag dim" style="margin:0;">NomaDamas · MinSync · agentdir · jikji</span>
  </div>
  <p style="margin-top:16pt;font-size:10pt;color:var(--ink);font-weight:800;text-align:center;">Thank you. Let's build the knowledge layer agents deserve.</p>
</div>
${footer('Journey of AutoRAG','21')}`));

slides.forEach((html, i) => {
  const name = `slide-${String(i + 1).padStart(2, '0')}.html`;
  writeFileSync(join(OUT, name), html, 'utf8');
});
console.log(`wrote ${slides.length} slides to ${OUT}`);
