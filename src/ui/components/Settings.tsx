import { useEffect, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Dialog from '@mui/material/Dialog';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import Divider from '@mui/material/Divider';
import IconButton from '@mui/material/IconButton';
import LinearProgress from '@mui/material/LinearProgress';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import Switch from '@mui/material/Switch';
import TextField from '@mui/material/TextField';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import Typography from '@mui/material/Typography';
import { formatDownload, useDownloadTracker } from '../downloadState';
import CloseIcon from '@mui/icons-material/Close';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import { api } from '../api';
import { ModelGate } from './ModelGate';
import { useColorMode } from '../colorMode';
import type { ThemeMode } from '../theme';
import type { ModelStatus } from '../../electron/ipc-types';

/**
 * How long the "context size saved" confirmation stays up. There is no
 * reload-finished signal from the backend (`model:set-context-size` kicks off
 * the reload and returns immediately), so this is a confirmation, not a status.
 */
const SAVED_NOTE_MS = 4000;

interface Props {
  status: ModelStatus;
  wordLevelEnabled: boolean;
  onWordLevelChange: (enabled: boolean) => void;
  onClose: () => void;
}

export function Settings({ status, wordLevelEnabled, onWordLevelChange, onClose }: Props) {
  const { mode, setMode } = useColorMode();
  const { active: download, cancel } = useDownloadTracker();
  const [contextDraft, setContextDraft] = useState('');
  const [savedNoteOpen, setSavedNoteOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getSettings().then((settings) => {
      setContextDraft(String(settings.contextSize));
    }).catch(() => { /* ignore */ });
  }, []);

  // The confirmation is dismissed here rather than by an animation-driven
  // auto-hide (Snackbar's `autoHideDuration` arms its timer in the enter
  // transition's `onEntered`): when the transition never runs — reduced motion,
  // a hidden window, jsdom — that timer never starts and the note sticks.
  useEffect(() => {
    if (!savedNoteOpen) return;
    const id = setTimeout(() => setSavedNoteOpen(false), SAVED_NOTE_MS);
    return () => clearTimeout(id);
  }, [savedNoteOpen]);

  async function handleSaveContextSize() {
    const parsed = Number(contextDraft);
    if (!Number.isInteger(parsed) || parsed < 256 || parsed > 131072) {
      setError('Context size must be an integer between 256 and 131072.');
      return;
    }
    setError(null);
    setSavedNoteOpen(false);
    try {
      const result = await api.setContextSize({ contextSize: parsed });
      setContextDraft(String(result.contextSize));
      setSavedNoteOpen(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <Dialog open onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ pr: 7 }}>Settings</DialogTitle>
      <IconButton
        onClick={onClose}
        aria-label="Close settings"
        title="Close settings"
        sx={{ position: 'absolute', right: 8, top: 8, color: 'text.secondary' }}
      >
        <CloseIcon />
      </IconButton>
      <Divider />
      <DialogContent dividers>
        {download && (
          <Alert
            severity="info"
            sx={{ mb: 2, alignItems: 'center' }}
            icon={<CircularProgress size={18} color="inherit" />}
            action={
              <Button size="small" color="inherit" onClick={() => void cancel()}>
                Cancel
              </Button>
            }
          >
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {formatDownload(download)}
            </Typography>
            <LinearProgress
              variant={download.total > 0 ? 'determinate' : 'indeterminate'}
              value={download.total > 0 ? download.percent : undefined}
              sx={{ mt: 1, height: 8, borderRadius: 4 }}
            />
          </Alert>
        )}

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
          General
        </Typography>
        <List disablePadding>
          <ListItem divider disableGutters>
            <ListItemText primary="Theme" secondary="Applies to the whole app" />
            <ToggleButtonGroup
              exclusive
              size="small"
              value={mode}
              onChange={(_, next) => {
                if (next === 'light' || next === 'dark') setMode(next as ThemeMode);
              }}
            >
              <ToggleButton value="light">
                <LightModeIcon fontSize="small" />
                <Box component="span" sx={{ ml: 0.75 }}>Light</Box>
              </ToggleButton>
              <ToggleButton value="dark">
                <DarkModeIcon fontSize="small" />
                <Box component="span" sx={{ ml: 0.75 }}>Dark</Box>
              </ToggleButton>
            </ToggleButtonGroup>
          </ListItem>
          <ListItem divider disableGutters>
            <ListItemText
              primary="Word-level corrections"
              secondary="Highlight misspelled words in the editor and offer one-click fixes"
            />
            <Switch
              checked={wordLevelEnabled}
              onChange={(event) => onWordLevelChange(event.target.checked)}
              slotProps={{ input: { 'aria-label': 'Word-level corrections' } }}
              title={wordLevelEnabled ? 'Enabled' : 'Disabled'}
            />
          </ListItem>
        </List>

        <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mt: 3 }}>
          LLM
        </Typography>
        <List disablePadding>
          <ListItem divider disableGutters sx={{ flexWrap: 'wrap' }}>
            <ListItemText
              primary="Context size (tokens)"
              secondary="Applies after the model reloads. Larger context = more memory."
            />
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexShrink: 0 }}>
              <TextField
                type="number"
                size="small"
                slotProps={{ htmlInput: { min: 256, max: 131072, step: 256 } }}
                value={contextDraft}
                onChange={(event) => {
                  setContextDraft(event.target.value);
                  setSavedNoteOpen(false);
                }}
                sx={{ width: 120 }}
              />
              <Button variant="contained" size="small" onClick={() => void handleSaveContextSize()}>
                Apply
              </Button>
            </Box>
            {savedNoteOpen && (
              <Alert
                severity="success"
                onClose={() => setSavedNoteOpen(false)}
                sx={{ mt: 1, width: '100%' }}
              >
                Saved — the new context size applies once the model has reloaded.
              </Alert>
            )}
          </ListItem>
        </List>

        <Box sx={{ mt: 3 }}>
          <ModelGate status={status} mode="manage" onClose={onClose} embedded />
        </Box>
      </DialogContent>
    </Dialog>
  );
}
