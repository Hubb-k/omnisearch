const SKIP_TAGS = new Set([
  'SCRIPT', 'STYLE', 'NAV', 'FOOTER', 'HEADER', 'ASIDE', 'NOSCRIPT',
  'SVG', 'PATH', 'DEFS', 'SYMBOL', 'USE', 'G', 'CIRCLE',
  'RECT', 'LINE', 'POLYGON', 'POLYLINE', 'ELLIPSE'
]);
const MAX_CHUNKS = 20;

function isJunk(chunk) {
  const junkRatio = (chunk.match(/[{}<>:;=\/\\]/g) || []).length / chunk.length;
  return junkRatio > 0.15;
}

function extractPageText() {
  function getTextNodes(node) {
    let text = '';
    for (const child of node.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) {
        text += child.textContent + ' ';
      } else if (child.nodeType === Node.ELEMENT_NODE && !SKIP_TAGS.has(child.tagName)) {
        text += getTextNodes(child);
      }
    }
    return text;
  }
  return getTextNodes(document.body || document.documentElement)
    .replace(/\s+/g, ' ').trim();
}

function sendToLocal(text, url, title) {
  const CHUNK = 500;
  const OVERLAP = 100;
  const chars = [...text];
  const chunks = [];

  for (let i = 0; i < chars.length; i += CHUNK - OVERLAP) {
    const chunk = chars.slice(i, i + CHUNK).join('').trim();
    if (chunk.length > 50 && !isJunk(chunk)) chunks.push(chunk);
    if (chunks.length >= MAX_CHUNKS) break;
  }

  if (chunks.length === 0) return;
  chrome.runtime.sendMessage({ type: 'INDEX_PAGE', chunks, url, title });
}

window.addEventListener('load', () => {
  setTimeout(() => {
    const text = extractPageText();
    if (text.length >= 500) {
      sendToLocal(text, location.href, document.title);
    }
  }, 1500);
});