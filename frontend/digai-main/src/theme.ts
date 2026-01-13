import { createTheme } from "@mui/material";
import { red } from "@mui/material/colors";

const theme = createTheme({
  palette: {
    primary: {
      main: "#f4623a",
    },
    secondary: {
      main: "#3accf4",
    },
    text: {
        primary: "#000000"
    },
    error: {
      main: red.A400,
    },
  },
});

export default theme;