import { createTheme } from '@mui/material/styles';

// Sci-Fi Color Palette
const colors = {
  darkPurple: '#1a0d2e',      // Deep purple background
  darkerPurple: '#0f0519',    // Even darker purple
  purple: '#4a148c',          // Medium purple
  lightPurple: '#6a1b9a',     // Lighter purple accents
  orange: '#ff6b35',          // Vibrant orange
  darkOrange: '#e55a2b',      // Darker orange
  accent: '#9d4edd',          // Purple-cyan accent (less harsh than neon blue)
  accentLight: '#c77dff',    // Lighter accent
  accentDark: '#7b2cbf',     // Darker accent
  darkGrey: '#1a1a1a',        // Dark grey
  black: '#000000',           // Pure black
  lightGrey: '#2d2d2d',       // Light grey for cards
  textPrimary: '#e0e0e0',     // Light text
  textSecondary: '#b0b0b0',   // Secondary text
};

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: colors.accent,
      light: colors.accentLight,
      dark: colors.accentDark,
      contrastText: colors.textPrimary,
    },
    secondary: {
      main: colors.orange,
      light: '#ff8c5a',
      dark: colors.darkOrange,
      contrastText: colors.black,
    },
    background: {
      default: colors.darkPurple,
      paper: colors.darkerPurple,
    },
    text: {
      primary: colors.textPrimary,
      secondary: colors.textSecondary,
    },
    error: {
      main: '#ff4444',
    },
    warning: {
      main: colors.orange,
    },
    info: {
      main: colors.accent,
    },
    success: {
      main: '#00ff88',
    },
  },
  typography: {
    // Use readable font for body text
    fontFamily: '"Roboto", "Inter", "Helvetica", "Arial", sans-serif',
    // Use Orbitron only for headings to maintain cyberpunk feel
    h1: {
      fontFamily: '"Orbitron", "Roboto", sans-serif',
      fontWeight: 700,
      letterSpacing: '0.03em',
      textTransform: 'uppercase',
    },
    h2: {
      fontFamily: '"Orbitron", "Roboto", sans-serif',
      fontWeight: 700,
      letterSpacing: '0.03em',
      textTransform: 'uppercase',
    },
    h3: {
      fontFamily: '"Orbitron", "Roboto", sans-serif',
      fontWeight: 600,
      letterSpacing: '0.02em',
    },
    h4: {
      fontFamily: '"Orbitron", "Roboto", sans-serif',
      fontWeight: 600,
      letterSpacing: '0.02em',
    },
    h5: {
      fontFamily: '"Orbitron", "Roboto", sans-serif',
      fontWeight: 600,
      letterSpacing: '0.01em',
    },
    h6: {
      fontFamily: '"Orbitron", "Roboto", sans-serif',
      fontWeight: 600,
    },
    body1: {
      fontFamily: '"Roboto", "Inter", sans-serif',
      letterSpacing: '0.01em',
      lineHeight: 1.6,
    },
    body2: {
      fontFamily: '"Roboto", "Inter", sans-serif',
      letterSpacing: '0.01em',
      lineHeight: 1.5,
    },
    button: {
      fontFamily: '"Orbitron", "Roboto", sans-serif',
      fontWeight: 600,
      letterSpacing: '0.05em',
      textTransform: 'uppercase',
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          background: `linear-gradient(135deg, ${colors.darkPurple} 0%, ${colors.darkerPurple} 100%)`,
          backgroundAttachment: 'fixed',
          '&::before': {
            content: '""',
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: `radial-gradient(circle at 20% 50%, ${colors.purple}15 0%, transparent 50%),
                         radial-gradient(circle at 80% 80%, ${colors.orange}10 0%, transparent 50%)`,
            pointerEvents: 'none',
            zIndex: 0,
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: `linear-gradient(135deg, ${colors.darkerPurple} 0%, ${colors.darkPurple} 100%)`,
          borderBottom: `2px solid ${colors.accent}40`,
          boxShadow: `0 4px 20px ${colors.accent}20, 0 0 40px ${colors.purple}20`,
          backdropFilter: 'blur(10px)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          background: `linear-gradient(135deg, ${colors.darkerPurple} 0%, ${colors.darkGrey} 100%)`,
          border: `1px solid ${colors.accent}30`,
          borderRadius: '12px',
          boxShadow: `0 8px 32px ${colors.black}80, 
                      0 0 20px ${colors.accent}10,
                      inset 0 1px 0 ${colors.accent}20`,
          transition: 'all 0.3s ease',
          '&:hover': {
            borderColor: `${colors.accent}60`,
            boxShadow: `0 12px 48px ${colors.black}90, 
                        0 0 30px ${colors.accent}20,
                        inset 0 1px 0 ${colors.accent}30`,
            transform: 'translateY(-2px)',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          padding: '10px 24px',
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
          transition: 'all 0.3s ease',
          '&.MuiButton-containedPrimary': {
            background: `linear-gradient(135deg, ${colors.accent} 0%, ${colors.accentDark} 100%)`,
            boxShadow: `0 4px 15px ${colors.accent}40, 0 0 20px ${colors.accent}20`,
            '&:hover': {
              background: `linear-gradient(135deg, ${colors.accentDark} 0%, ${colors.accent} 100%)`,
              boxShadow: `0 6px 20px ${colors.accent}60, 0 0 30px ${colors.accent}30`,
              transform: 'translateY(-2px)',
            },
          },
          '&.MuiButton-containedSecondary': {
            background: `linear-gradient(135deg, ${colors.orange} 0%, ${colors.darkOrange} 100%)`,
            boxShadow: `0 4px 15px ${colors.orange}40, 0 0 20px ${colors.orange}20`,
            '&:hover': {
              background: `linear-gradient(135deg, ${colors.darkOrange} 0%, ${colors.orange} 100%)`,
              boxShadow: `0 6px 20px ${colors.orange}60, 0 0 30px ${colors.orange}30`,
              transform: 'translateY(-2px)',
            },
          },
          '&.MuiButton-outlined': {
            borderWidth: '2px',
            '&.MuiButton-outlinedPrimary': {
              borderColor: colors.accent,
              color: colors.accent,
              '&:hover': {
                borderColor: colors.accent,
                background: `${colors.accent}15`,
                boxShadow: `0 0 20px ${colors.accent}30`,
              },
            },
            '&.MuiButton-outlinedSecondary': {
              borderColor: colors.orange,
              color: colors.orange,
              '&:hover': {
                borderColor: colors.orange,
                background: `${colors.orange}15`,
                boxShadow: `0 0 20px ${colors.orange}30`,
              },
            },
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: '8px',
            '& fieldset': {
              borderColor: `${colors.accent}40`,
              borderWidth: '2px',
            },
            '&:hover fieldset': {
              borderColor: `${colors.accent}60`,
            },
            '&.Mui-focused fieldset': {
              borderColor: colors.accent,
              boxShadow: `0 0 15px ${colors.accent}30`,
            },
            '& input': {
              color: colors.textPrimary,
            },
            '& textarea': {
              color: colors.textPrimary,
            },
          },
          '& .MuiInputLabel-root': {
            color: colors.textSecondary,
            '&.Mui-focused': {
              color: colors.accent,
            },
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: '6px',
          fontWeight: 600,
          '&.MuiChip-colorPrimary': {
            background: `${colors.accent}20`,
            color: colors.accent,
            border: `1px solid ${colors.accent}40`,
          },
          '&.MuiChip-colorSecondary': {
            background: `${colors.orange}20`,
            color: colors.orange,
            border: `1px solid ${colors.orange}40`,
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          background: `linear-gradient(135deg, ${colors.darkerPurple} 0%, ${colors.darkGrey} 100%)`,
          border: `1px solid ${colors.accent}20`,
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          borderRadius: '4px',
          backgroundColor: `${colors.darkGrey}`,
          '& .MuiLinearProgress-bar': {
            background: `linear-gradient(90deg, ${colors.accent} 0%, ${colors.orange} 100%)`,
            boxShadow: `0 0 10px ${colors.accent}50`,
          },
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          border: '1px solid',
          '&.MuiAlert-standardInfo': {
            borderColor: `${colors.accent}40`,
            background: `${colors.accent}10`,
            color: colors.accent,
          },
          '&.MuiAlert-standardSuccess': {
            borderColor: '#00ff8840',
            background: '#00ff8810',
            color: '#00ff88',
          },
          '&.MuiAlert-standardWarning': {
            borderColor: `${colors.orange}40`,
            background: `${colors.orange}10`,
            color: colors.orange,
          },
          '&.MuiAlert-standardError': {
            borderColor: '#ff444440',
            background: '#ff444410',
            color: '#ff4444',
          },
        },
      },
    },
    MuiMenuItem: {
      styleOverrides: {
        root: {
          '&:hover': {
            background: `${colors.accent}20`,
          },
          '&.Mui-selected': {
            background: `${colors.accent}30`,
            '&:hover': {
              background: `${colors.accent}40`,
            },
          },
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          background: `linear-gradient(135deg, ${colors.darkerPurple} 0%, ${colors.darkGrey} 100%)`,
          border: `2px solid ${colors.accent}40`,
          boxShadow: `0 20px 60px ${colors.black}90, 0 0 40px ${colors.accent}20`,
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        root: {
          borderBottom: `2px solid ${colors.accent}30`,
        },
        indicator: {
          background: colors.accent,
          boxShadow: `0 0 10px ${colors.accent}50`,
          height: '3px',
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          fontFamily: '"Orbitron", "Roboto", sans-serif',
          color: colors.textSecondary,
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          '&.Mui-selected': {
            color: colors.accent,
          },
          '&:hover': {
            color: colors.accent,
          },
        },
      },
    },
    MuiTypography: {
      styleOverrides: {
        root: {
          '&.MuiTypography-body1, &.MuiTypography-body2, &.MuiTypography-caption, &.MuiTypography-overline': {
            fontFamily: '"Roboto", "Inter", sans-serif',
          },
        },
      },
    },
  },
});

export default theme;

