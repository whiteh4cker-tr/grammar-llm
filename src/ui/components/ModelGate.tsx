import { useEffect, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import IconButton from '@mui/material/IconButton';
import LinearProgress from '@mui/material/LinearProgress';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import Radio from '@mui/material/Radio';
import RadioGroup from '@mui/material/RadioGroup';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { alpha, useTheme } from '@mui/material/styles';
import { SUCCESS_TEXT } from '../theme';
import CloseIcon from '@mui/icons-material/Close';
import DeleteIcon from '@mui/icons-material/Delete';
import DownloadIcon from '@mui/icons-material/Download';
import { api } from '../api';
import { formatDownload, useDownloadTracker } from '../downloadState';
import type { ModelStatus } from '../../electron/ipc-types';

const MODELS = [
  {
    fileName: 'GRMR-V3-G4B-Q4_K_M.gguf',
    url: 'https://huggingface.co/icecubetr/GRMR-V3-G4B-GGUF/resolve/main/GRMR-V3-G4B-Q4_K_M.gguf',
    label: 'GRMR-V3-G4B-Q4_K_M',
    detail: 'Recommended — faster, smaller download',
  },
  {
    fileName: 'GRMR-V3-G4B-Q8_0.gguf',
    url: 'https://huggingface.co/icecubetr/GRMR-V3-G4B-GGUF/resolve/main/GRMR-V3-G4B-Q8_0.gguf',
    label: 'GRMR-V3-G4B-Q8_0',
    detail: 'Highest quality — slower, larger download',
  },
];

function fileNameFromUrl(url: string): string | null {
  try {
    const parsed = new URL(url);
    const segment = parsed.pathname.split('/').filter(Boolean).pop();
    if (!segment) return null;
    return decodeURIComponent(segment);
  } catch {
    return null;
  }
}

interface Props {
  status: ModelStatus;
  mode?: 'required' | 'manage';
  onClose?: () => void;
  embedded?: boolean;
}

export function ModelGate({ status, mode = 'required', onClose, embedded = false }: Props) {
  const theme = useTheme();
  const [selected, setSelected] = useState(0);
  const [customUrl, setCustomUrl] = useState('');
  const [installed, setInstalled] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  // Download state lives above this component (it unmounts when the app
  // switches screens or the dialog closes) — see `DownloadProvider`.
  const { active: progress, track, cancel } = useDownloadTracker();
  const downloading = progress !== null;

  // Re-read the folder when the download finishes so the new file appears.
  useEffect(() => {
    let cancelled = false;
    api.listModels().then((models) => {
      if (!cancelled) setInstalled(models);
    }).catch(() => { /* ignore */ });
    return () => { cancelled = true; };
  }, [mode, downloading]);

  const isCustom = selected === MODELS.length;
  const customFileName = isCustom ? fileNameFromUrl(customUrl) : null;
  const customValid = isCustom && customUrl.trim().length > 0 && customFileName !== null && customFileName.endsWith('.gguf');

  const chosen = isCustom
    ? { fileName: customFileName, url: customUrl.trim() }
    : MODELS[selected];

  const buttonLabel = isCustom
    ? (customValid ? `Download ${customFileName}` : 'Enter a valid .gguf URL')
    : (installed.includes(MODELS[selected].fileName)
        ? `Use ${MODELS[selected].label}`
        : `Download ${MODELS[selected].label}`);

  async function handlePrimaryAction() {
    const fileName = chosen.fileName;
    if (!fileName) return;
    setError(null);

    // Already installed → just switch to it, no re-download.
    if (installed.includes(fileName)) {
      try {
        await api.selectModel({ fileName });
        if (mode === 'manage' && onClose) onClose();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
      return;
    }

    try {
      await track(fileName, () => api.downloadModel({ url: chosen.url, fileName }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function handleSelect(fileName: string) {
    setError(null);
    try {
      await api.selectModel({ fileName });
      if (onClose) onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function handleDelete(fileName: string) {
    setError(null);
    try {
      const next = await api.deleteModel({ fileName });
      setInstalled((prev) => prev.filter((name) => name !== fileName));
      if (next.state === 'missing' && mode === 'manage' && onClose) {
        onClose();
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function handleCancel() {
    await cancel();
  }

  /** Selected-vs-idle look for the radio option cards. */
  const optionSx = (isSelected: boolean) => ({
    borderColor: isSelected ? 'primary.main' : 'divider',
    bgcolor: isSelected ? alpha(theme.palette.primary.main, 0.08) : 'background.paper',
    '&:hover': { borderColor: 'primary.main' },
  } as const);

  const content = (
    <>
      {!embedded && (
        <Box sx={{ position: 'relative', textAlign: 'center', mb: 1 }}>
          <Typography variant="h4" component="h1">GrammarLLM</Typography>
          {mode === 'manage' && onClose && (
            <IconButton onClick={onClose} aria-label="Close" sx={{ position: 'absolute', right: -8, top: 0 }}>
              <CloseIcon />
            </IconButton>
          )}
        </Box>
      )}

      <Typography
        variant="body2"
        color="text.secondary"
        sx={{ mb: 3, textAlign: embedded ? 'left' : 'center' }}
      >
        {mode === 'manage'
          ? 'Manage your models — switch, download, or delete.'
          : 'No model detected. Choose a model to download and get started.'}
      </Typography>

      {installed.length > 0 && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="overline" color="text.secondary" sx={{ display: 'block' }}>
            Installed models
          </Typography>
          <List disablePadding>
            {installed.map((name) => {
              const isActive = status.modelName === name && status.state === 'ready';
              return (
                <ListItem key={name} divider disableGutters sx={{ py: 1, gap: 1 }}>
                  <ListItemText
                    primary={name}
                    slotProps={{
                      primary: {
                        noWrap: true,
                        title: name,
                        // `success.main` is a fill colour; as text it is too dark
                        // for the dark surface it sits on.
                        sx: isActive ? { color: ({ palette }) => SUCCESS_TEXT[palette.mode], fontWeight: 600 } : undefined,
                      },
                    }}
                  />
                  {isActive && <Chip size="small" color="success" variant="outlined" label="in use" />}
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexShrink: 0 }}>
                    {!isActive && (
                      <Button size="small" variant="outlined" onClick={() => void handleSelect(name)}>
                        Use
                      </Button>
                    )}
                    <Button
                      size="small"
                      color="error"
                      startIcon={<DeleteIcon fontSize="small" />}
                      onClick={() => void handleDelete(name)}
                    >
                      Delete
                    </Button>
                  </Box>
                </ListItem>
              );
            })}
          </List>
        </Box>
      )}

      <RadioGroup
        value={String(selected)}
        onChange={(event) => {
          if (event.target.value !== '') setSelected(Number(event.target.value));
        }}
        sx={{ gap: 1.5, mb: 3 }}
      >
        {MODELS.map((model, index) => (
          // The whole option card is a <label>, so clicking anywhere on it
          // selects the radio (no nested <label>, which would be invalid).
          <Card
            key={model.fileName}
            component="label"
            variant="outlined"
            sx={{ ...optionSx(selected === index), cursor: downloading ? 'default' : 'pointer' }}
          >
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, px: 2, py: 1.25 }}>
              <Radio value={String(index)} disabled={downloading} sx={{ py: 0.5 }} />
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.25 }}>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>{model.label}</Typography>
                <Typography variant="caption" color="text.secondary">{model.detail}</Typography>
                {installed.includes(model.fileName) && (
                  <Typography variant="caption" sx={{ color: ({ palette }) => SUCCESS_TEXT[palette.mode] }}>
                    already installed
                  </Typography>
                )}
              </Box>
            </Box>
          </Card>
        ))}

        <Card variant="outlined" sx={{ ...optionSx(isCustom), px: 2, py: 1.25 }}>
          <Box component="label" sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, cursor: downloading ? 'default' : 'pointer' }}>
            <Radio value={String(MODELS.length)} disabled={downloading} sx={{ py: 0.5 }} />
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.25 }}>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>Custom GGUF URL</Typography>
              <Typography variant="caption" color="text.secondary">
                Paste a direct download link to any .gguf model (e.g., from Hugging Face)
              </Typography>
            </Box>
          </Box>
          {isCustom && (
            <TextField
              fullWidth
              size="small"
              type="url"
              placeholder="https://huggingface.co/.../model.gguf"
              value={customUrl}
              onChange={(event) => setCustomUrl(event.target.value)}
              disabled={downloading}
              sx={{ mt: 1, ml: 4.5, width: 'calc(100% - 36px)' }}
            />
          )}
        </Card>
      </RadioGroup>

      {status.state === 'error' && (
        <Alert severity="error" sx={{ mb: 2 }}>Model failed to load: {status.modelName}</Alert>
      )}
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {progress ? (
        <Box sx={{ mt: 1, textAlign: 'left' }}>
          <Typography variant="body2" sx={{ mb: 1 }} color="text.secondary">
            {formatDownload(progress)}
          </Typography>
          <LinearProgress
            variant={progress.total > 0 ? 'determinate' : 'indeterminate'}
            value={progress.total > 0 ? progress.percent : undefined}
            sx={{ height: 10, borderRadius: 5, bgcolor: 'action.disabledBackground' }}
          />
          <Button variant="outlined" color="error" fullWidth sx={{ mt: 2 }} onClick={() => void handleCancel()}>
            Cancel
          </Button>
        </Box>
      ) : (
        <Button
          variant="contained"
          size="large"
          fullWidth
          onClick={() => void handlePrimaryAction()}
          disabled={downloading || (isCustom && !customValid)}
          startIcon={downloading ? <CircularProgress size={18} color="inherit" /> : <DownloadIcon />}
        >
          {buttonLabel}
        </Button>
      )}
    </>
  );

  if (embedded) {
    return <Box sx={{ pt: 1 }}>{content}</Box>;
  }

  return (
    <Box sx={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', p: 3 }}>
      <Card elevation={8} sx={{ width: '100%', maxWidth: 560, p: { xs: 2.5, sm: 4 }, borderRadius: 2.5 }}>
        {content}
      </Card>
    </Box>
  );
}
