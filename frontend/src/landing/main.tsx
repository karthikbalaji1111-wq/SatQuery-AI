/**
 * Separate entry point for the landing page.
 *
 * A distinct Vite entry rather than a route inside the app: the workspace is
 * frozen, and this way nothing in it is touched - not its bundle, not its
 * state, not its stylesheet. "Try SatQuery" is a plain link to `/`, which is
 * the real application.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { Landing } from "./Landing";
import "./landing.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Landing />
  </StrictMode>,
);
