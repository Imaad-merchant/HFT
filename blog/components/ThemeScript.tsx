// Runs before first paint so dark mode doesn't flash on reload.
export default function ThemeScript() {
  const code = `
(function () {
  try {
    var stored = localStorage.getItem('theme');
    var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    var dark = stored ? stored === 'dark' : prefersDark;
    if (dark) document.documentElement.classList.add('dark');
  } catch (e) {}
})();
`;
  return <script dangerouslySetInnerHTML={{ __html: code }} />;
}
