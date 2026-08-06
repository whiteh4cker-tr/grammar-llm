import { useEffect, useState } from 'react';

export function ThemeToggle() {
  const [dark, setDark] = useState(() => {
    try {
      return localStorage.getItem('theme') !== 'light';
    } catch {
      return true;
    }
  });

  useEffect(() => {
    document.body.classList.toggle('dark-mode', dark);
    try {
      localStorage.setItem('theme', dark ? 'dark' : 'light');
    } catch {
      // ignore storage errors
    }
  }, [dark]);

  return (
    <button
      className="theme-toggle"
      onClick={() => setDark((d) => !d)}
      title={dark ? 'Dark Mode (Click to switch to Light Mode)' : 'Light Mode (Click to switch to Dark Mode)'}
    >
      {dark ? '🌙' : '☀️'}
    </button>
  );
}
