const WS_URL = 'ws://127.0.0.1:45678';
const SEEN_URLS = new Set();
let ws = null;
let queue = [];

function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    while (queue.length > 0) ws.send(JSON.stringify(queue.shift()));
  };

  ws.onclose = () => {
    ws = null;
    setTimeout(connect, 5000);
  };

  ws.onerror = () => {};
}

function send(data) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(data));
  } else {
    queue.push(data);
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
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'ping' }));
  } else {
    connect();
  }
}, 20000);