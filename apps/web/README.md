# RetentionIQ web app

The interactive decision layer: Customer Intelligence, Customer Risk, Model Lab, Experiment Lab,
Uplift Lab and Campaign Simulator. Every number comes from the RetentionIQ API, which serves the
`make evaluate` artifacts.

```bash
npm install
RIQ_API_URL=http://localhost:8000 npm run dev   # http://localhost:3000
npm run build && npm start                       # production
```

Charts use the validated palette tokens in `app/globals.css` (light and dark).
