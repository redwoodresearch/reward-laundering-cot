// Runs every figure script of a built figure page against a minimal DOM stub and reports each figure's SVG size or
// error, so a broken figure is caught without a browser:  node analysis/check_figures.js figures.html
const fs = require("fs"), vm = require("vm");
const page = fs.readFileSync(process.argv[2], "utf8");
const scripts = [...page.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
let failed = 0;
for (const code of scripts) {
  const hosts = {};
  const document = {
    getElementById: id => (hosts[id] = hosts[id] || { id, innerHTML: "", appendChild() {} }),
    createElement: () => ({ getContext: () => null }),
    addEventListener() {},
  };
  try {
    vm.runInNewContext(code, { document, console, URL: undefined, Blob: undefined, Image: undefined });
  } catch (e) {
    failed++;
    console.log(`ERROR in script starting ${JSON.stringify(code.slice(0, 60))}: ${e.message}`);
    continue;
  }
  for (const [id, h] of Object.entries(hosts)) {
    const bad = /NaN|undefined/.test(h.innerHTML);
    if (bad) failed++;
    console.log(`${id}: ${h.innerHTML.length} chars of SVG${bad ? " — contains NaN/undefined" : ""}`);
  }
}
process.exit(failed ? 1 : 0);
