# Signal Trace browser extension (Chrome and Edge)

This optional Manifest V3 extension checks URLs against the API's configured local exact-match IOC dataset. The included dataset is synthetic and suitable only for development. A match is a dataset lookup result, not an independent finding of real-world maliciousness.

## Install locally

1. Start the FastAPI service from the project root: `python -m uvicorn app.main:app --reload`.
2. In Chrome or Edge, open the browser's extensions page and enable Developer mode.
3. Choose **Load unpacked** and select the project's `browser_extension` directory.
4. Pin the extension if desired. Use its toolbar action to check the current page, or right-click a link and choose **Check link with Signal Trace**.
5. Grant optional page or download permissions only if you want those background checks. Each can be disabled from the popup.

## Permissions and limits

- The API host permission is limited to `127.0.0.1:8000` and `localhost:8000`; the extension sends no data to a remote service.
- Manual page/link checks use the URL supplied by the user and do not persist it as a session. Exact-match checks are read-only unless an API caller explicitly supplies a session ID.
- Optional page monitoring requires `webNavigation` and `http://*/*` plus `https://*/*` host access. It checks top-level HTTP/HTTPS navigations and can show an OS/browser notification for a local IOC match. The extension does not store browsing history.
- Browser navigation events arrive after navigation has started. This permitted browser API does not reliably block a page before it loads, so this feature is a notification, not a blocking safe-browsing system. Browser internal pages and restricted schemes are skipped.
- Optional download monitoring requires the browser's `downloads` permission. It checks the download source URL when the browser reports a new download. It does not inspect, open, scan, quarantine, or delete the downloaded file.
- Notifications depend on browser and operating-system notification settings. The local API must be running for checks to work.
- QR camera capture is provided by the dashboard, not the extension. The dashboard requests camera access only after the user selects **Open camera**, stops the stream after capture/close, and submits the captured still image to the existing in-memory QR analyzer. Cameras generally require HTTPS or `localhost`/a loopback origin.
- Mobile browser extension support varies by browser and OS. On mobile, use the responsive dashboard and device camera/file picker; this project does not install an OS-level interception service.
- Email remains website-only. No mailbox API, mail-client add-in, or background email access is included.
