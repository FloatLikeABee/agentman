import React, { useState } from 'react';
import {
  TextField,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  InputAdornment,
} from '@mui/material';
import {
  OpenInFull as OpenInFullIcon,
} from '@mui/icons-material';

const SystemPromptInput = ({
  label = "System Prompt",
  value,
  onChange,
  placeholder,
  helperText,
  rows = 4,
  fullWidth = true,
  required = false,
  ...otherProps
}) => {
  const [openModal, setOpenModal] = useState(false);
  const [modalValue, setModalValue] = useState(value || '');

  const handleOpenModal = () => {
    setModalValue(value || '');
    setOpenModal(true);
  };

  const handleCloseModal = () => {
    setOpenModal(false);
  };

  const handleSaveModal = () => {
    onChange({ target: { value: modalValue } });
    setOpenModal(false);
  };

  // Update modal value when external value changes
  React.useEffect(() => {
    if (!openModal) {
      setModalValue(value || '');
    }
  }, [value, openModal]);

  return (
    <>
      <Box sx={{ position: 'relative', width: fullWidth ? '100%' : 'auto' }}>
        <TextField
          {...otherProps}
          fullWidth={fullWidth}
          multiline
          rows={rows}
          label={label}
          value={value || ''}
          onChange={onChange}
          placeholder={placeholder}
          helperText={helperText}
          required={required}
          InputProps={{
            endAdornment: (
              <InputAdornment position="end" sx={{ alignSelf: 'flex-start', mt: 1 }}>
                <IconButton
                  size="small"
                  onClick={handleOpenModal}
                  title="Open in larger editor"
                  onMouseDown={(e) => e.preventDefault()} // Prevent focus on TextField
                >
                  <OpenInFullIcon fontSize="small" />
                </IconButton>
              </InputAdornment>
            ),
          }}
          sx={otherProps.sx}
        />
      </Box>

      <Dialog
        open={openModal}
        onClose={handleCloseModal}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: {
            height: '80vh',
            maxHeight: '80vh',
          },
        }}
      >
        <DialogTitle>
          {label}
          {required && <span style={{ color: 'red', marginLeft: 4 }}>*</span>}
        </DialogTitle>
        <DialogContent sx={{ display: 'flex', flexDirection: 'column', p: 2 }}>
          <TextField
            fullWidth
            multiline
            value={modalValue}
            onChange={(e) => setModalValue(e.target.value)}
            placeholder={placeholder}
            helperText={helperText}
            sx={{
              flex: 1,
              '& .MuiInputBase-root': {
                height: '100%',
                alignItems: 'flex-start',
              },
              '& textarea': {
                height: 'calc(80vh - 200px) !important',
                overflow: 'auto !important',
                resize: 'none',
              },
            }}
            inputProps={{
              style: {
                height: '100%',
                overflow: 'auto',
              },
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseModal}>Cancel</Button>
          <Button onClick={handleSaveModal} variant="contained">
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default SystemPromptInput;
