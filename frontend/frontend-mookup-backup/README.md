# AI Telephony Platform Frontend

This project contains a high-fidelity mockup of a B2B AI telephony control center built with React, TypeScript, and Vite. It showcases all 25 pages outlined in the product requirements, complete with sample data, navigation, and reusable UI patterns inspired by the ShadCN design language.

## Getting started

```bash
npm install
npm run dev
```

The application is desktop-first. Routes are handled via React Router and include dedicated pages for establishments, agents, calls, reports, settings, and help resources. React Query and Zustand are configured to illustrate the state management approach for future data integrations.

## Available scripts

- `npm run dev` – start the Vite dev server
- `npm run build` – type-check and build the production bundle
- `npm run preview` – preview the production build
- `npm run lint` – run TypeScript in noEmit mode

## Project structure

```
src/
  components/      # Layout, reusable UI
  data/            # Mock data for cards, tables, charts
  pages/           # 25 page-level components matching the PRD
  stores/          # Zustand store examples
```

The styling is implemented with handcrafted CSS tokens for quick iteration and to mirror the tonal qualities of the intended design system (primary indigo, deep surfaces, and soft gradients).
