const WS_URL = 'ws://127.0.0.1:45678';
const SEEN_URLS = new Set();
let ws = null;
let queue = [];
let retryDelay = 1000;
let retryTimer = null;

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    retryDelay = 1000;
    while (queue.length > 0) ws.send(JSON.stringify(queue.shift()));
  };

  ws.onclose = () => {
    ws = null;
    scheduleReconnect();
  };

  ws.onerror = () => {};
}

function scheduleReconnect() {
  if (retryTimer) return;
  retryTimer = setTimeout(() => {
    retryTimer = null;
    retryDelay = Math.min(retryDelay * 2, 30000);
    connect();
  }, retryDelay);
}

function send(data) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(data));
  } else {
    queue.push(data);
    connect();
  }
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type !== 'INDEX_PAGE') return;
  if (SEEN_URLS.has(msg.url)) return;
  SEEN_URLS.add(msg.url);
  msg.chunks.forEach(chunk => {
    send({ type: 'index', text: chunk, source: msg.url, title: msg.title });
  });
});

connect();

setInterval(() => {
  if (!ws || ws.readyState !== WebSocket.OPEN) connect();
}, 20000);