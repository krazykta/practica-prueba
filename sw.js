const CACHE = "practica-v1-8-11";
const ASSETS = [
  "./",
  "./index.html",
  "./manifest.json",
  "./separator-worker.js",
  "./lame.min.js",
  "./fonts/manrope.woff2",
  "./icons/icon-180.png",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
  "./icons/icon-maskable-512.png"
];

self.addEventListener("install", event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener("activate", event => {
  event.waitUntil(
    // Solo cachés antiguas de esta app: no tocar el modelo de IA (practica-modelos-*) ni otras apps del mismo dominio.
    caches.keys().then(keys => Promise.all(keys.filter(k => k.startsWith("practica-v") && k !== CACHE).map(k => caches.delete(k))))
  );
  self.clients.claim();
});

// Tocar el aviso "Práctica en curso" vuelve a la app.
self.addEventListener("notificationclick", event => {
  event.notification.close();
  event.waitUntil(clients.matchAll({ type: "window", includeUncontrolled: true })
    .then(list => list.length ? list[0].focus() : clients.openWindow("./")));
});

self.addEventListener("fetch", event => {
  if (event.request.method !== "GET") return;

  // Authentication endpoints must always bypass the PWA cache.
  const url = new URL(event.request.url);
  // El modelo de IA (57 MB) ya lo cachea el worker; no duplicarlo aquí.
  if (/(^|\.)(huggingface\.co|hf\.co)$/.test(url.hostname)) return;
  if (url.origin === self.location.origin && url.pathname.startsWith("/cdn-cgi/access/")) {
    event.respondWith(fetch(event.request));
    return;
  }

  // Always prefer the newest app shell when online, but keep an offline fallback.
  if (event.request.mode === "navigate") {
    event.respondWith(
      fetch(event.request)
        .then(response => {
          const copy = response.clone();
          caches.open(CACHE).then(cache => cache.put("./index.html", copy));
          return response;
        })
        .catch(() => caches.match("./index.html"))
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then(cached => {
      const fresh = fetch(event.request).then(response => {
        if (response && response.ok) {
          const copy = response.clone();
          caches.open(CACHE).then(cache => cache.put(event.request, copy));
        }
        return response;
      }).catch(() => cached);
      return cached || fresh;
    })
  );
});
