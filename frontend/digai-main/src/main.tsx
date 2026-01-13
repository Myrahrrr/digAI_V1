import React from "react";
import ReactDOM from "react-dom/client";
import Home from "./pages/Home.tsx";
import "./index.css";
import { CssBaseline, ThemeProvider } from "@mui/material";
import theme from "./theme.ts";
import { RouterProvider, createBrowserRouter } from "react-router-dom";
import PartA from "./pages/PartA.tsx";
import PartB from "./pages/PartB.tsx";
import InterAB from "./pages/InterAB.tsx";
import InterBC from "./pages/InterBC.tsx";
import PartC from "./pages/PartC.tsx";
import End from "./pages/End.tsx";
import Questions from "./pages/Questions.tsx";
import Instructions from "./pages/Instructions.tsx";
import "./i18n";
import Terms from "./pages/Terms.tsx";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Home />,
  },
  {
    path: "/part-a",
    element: <PartA />,
  },
  {
    path: "/inter-ab",
    element: <InterAB />,
  },
  {
    path: "/part-b",
    element: <PartB />,
  },
  {
    path: "/questions",
    element: <Questions />,
  },
  {
    path: "/inter-bc",
    element: <InterBC />,
  },
  {
    path: "/part-c",
    element: <PartC />,
  },
  {
    path: "/end",
    element: <End />,
  },
  {
    path: "/instructions",
    element: <Instructions />,
  },
  {
    path: "/terms",
    element: <Terms />,
  },
]);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <RouterProvider router={router} />
    </ThemeProvider>
  </React.StrictMode>
);
