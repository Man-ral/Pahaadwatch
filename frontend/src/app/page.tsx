"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";

const LocationMap = dynamic(() => import("@/components/LocationMap"), {
  ssr: false,
  loading: () => <div className="map-loading">Loading map…</div>,
});

type Point = { latitude: number; longitude: number };
type SearchResult = { place_id: number; display_name: string; lat: string; lon: string };
type FeatureRecord = Record<string, number | string>;
type TemporalPrediction = {
  experimental_score: number;
  threshold: number;
  above_threshold?: boolean;
  prediction_horizon?: string;
  rainfall?: FeatureRecord;
  terrain?: FeatureRecord;
  notice?: string;
};
type SusceptibilityPrediction = {
  model_score: number;
  susceptibility_class: string;
  warning?: string;
  metadata?: Record<string, string | number>;
};
type Theme = "light" | "dark";
type SignalBand = {
  key: "low" | "elevated" | "high" | "very-high";
  label: string;
  explanation: string;
};

const INITIAL_POINT: Point = { latitude: 30.7268, longitude: 78.4354 };
const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? (process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : "")).replace(/\/$/, "");
const SAVED_EXAMPLE: TemporalPrediction = {
  experimental_score: 0.5933,
  threshold: 0.175,
  above_threshold: true,
  prediction_horizon: "next_3_days",
  rainfall: {
    RAIN_1D: 0, RAIN_3D: 0, RAIN_7D: 0, RAIN_15D: 26.4, RAIN_30D: 186.8,
    rainfall_source: "IMERG Early Run", rainfall_data_through: "2026-10-23", reference_date: "2026-10-24",
  },
  terrain: { ELEVATION: 1130, SLOPE: 10.5816, LANDCOVER_CODE: 30 },
  notice: "Saved example only. Not a live prediction or calibrated probability.",
};

function getSignalBand(score: number): SignalBand {
  if (score < 0.50) {
    return {
      key: "low",
      label: "Low Signal",
      explanation: "The model shows a lower risk signal.",
    };
  }

  if (score < 0.75) {
    return {
      key: "elevated",
      label: "Slightly Raised Signal",
      explanation: "The model shows a slightly raised risk signal.",
    };
  }

  if (score < 0.92) {
    return {
      key: "high",
      label: "High Signal",
      explanation: "The model shows a higher risk signal.",
    };
  }

  return {
    key: "very-high",
    label: "Very High Signal",
    explanation: "The model shows a very high risk signal.",
  };
}

function prettyDate(value: unknown) {
  if (typeof value !== "string") return "Date unavailable";
  const date = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-IN", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }).format(date);
}

function classKey(value: string) {
  return value.toLowerCase().replace(/[^a-z]+/g, "-").replace(/^-|-$/g, "");
}

export default function Home() {
  const [theme, setTheme] = useState<Theme>("light");
  const [selected, setSelected] = useState<Point>(INITIAL_POINT);
  const [mapCenter, setMapCenter] = useState<Point>(INITIAL_POINT);
  const [placeName, setPlaceName] = useState("Uttarkashi, Uttarakhand");
  const [query, setQuery] = useState("Uttarkashi, Uttarakhand");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState("");
  const [temporal, setTemporal] = useState<TemporalPrediction | null>(null);
  const [temporalKind, setTemporalKind] = useState<"live" | "example" | null>(null);
  const [temporalBusy, setTemporalBusy] = useState(false);
  const [temporalError, setTemporalError] = useState("");
  const [susceptibility, setSusceptibility] = useState<SusceptibilityPrediction | null>(null);
  const [susceptibilityBusy, setSusceptibilityBusy] = useState(false);
  const [susceptibilityError, setSusceptibilityError] = useState("");
  const [showTemporalData, setShowTemporalData] = useState(false);
  const [showSusceptibilityData, setShowSusceptibilityData] = useState(false);
  const [showProjectDetails, setShowProjectDetails] = useState(false);
  const lastSearchAt = useRef(0);

  useEffect(() => {
  let saved: "light" | "dark" | null = null;

  try {
    const value = window.localStorage.getItem("pahaadwatch-theme");

    if (value === "light" || value === "dark") {
      saved = value;
    }
  } catch {
    // Local storage may be unavailable.
  }

  if (
    !saved &&
    window.matchMedia?.("(prefers-color-scheme: dark)").matches
  ) {
    saved = "dark";
  }

  if (!saved) return;

  const frame = window.requestAnimationFrame(() => {
    setTheme(saved!);
  });

  return () => window.cancelAnimationFrame(frame);
}, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { window.localStorage.setItem("pahaadwatch-theme", theme); } catch { /* optional preference */ }
  }, [theme]);

  function clearResults() {
    setTemporal(null); setTemporalKind(null); setTemporalError("");
    setSusceptibility(null); setSusceptibilityError("");
    setShowTemporalData(false); setShowSusceptibilityData(false);
  }

  function selectPoint(point: Point, name?: string, recenter = false) {
    setSelected(point);
    setPlaceName(name ? name.split(",").slice(0, 2).join(",") : "Selected map point");
    if (name) setQuery(name.split(",").slice(0, 2).join(","));
    if (recenter) setMapCenter(point);
    clearResults();
  }

  async function searchPlace(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const term = query.trim();
    if (term.length < 3) { setSearchError("Enter a town, village or place name."); return; }
    setSearching(true); setSearchError(""); setResults([]);
    try {
      const elapsed = Date.now() - lastSearchAt.current;
      if (lastSearchAt.current && elapsed < 1100) await new Promise((resolve) => setTimeout(resolve, 1100 - elapsed));
      lastSearchAt.current = Date.now();
      const params = new URLSearchParams({ format: "jsonv2", q: term, limit: "5", countrycodes: "in", viewbox: "75.4,33.4,81.2,28.7", bounded: "1" });
      const response = await fetch(`https://nominatim.openstreetmap.org/search?${params.toString()}`, { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("Place search is unavailable right now. Try choosing the point on the map.");
      const data = (await response.json()) as SearchResult[];
      if (!data.length) { setSearchError("No matching place found. Try a nearby town or tap the map."); return; }
      setResults(data);
    } catch (error) {
      setSearchError(error instanceof Error ? error.message : "Place search is unavailable right now.");
    } finally { setSearching(false); }
  }

  function chooseSearchResult(result: SearchResult) {
    selectPoint({ latitude: Number(result.lat), longitude: Number(result.lon) }, result.display_name, true);
    setResults([]); setSearchError("");
  }

  function useMyLocation() {
    setSearchError("");
    if (!navigator.geolocation) { setSearchError("This browser does not support location access."); return; }
    navigator.geolocation.getCurrentPosition(
      (position) => selectPoint({ latitude: Number(position.coords.latitude.toFixed(6)), longitude: Number(position.coords.longitude.toFixed(6)) }, "My current location", true),
      () => setSearchError("Location permission was not granted. Search for a place or tap the map instead."),
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 },
    );
  }

  async function requestModel(path: string) {
    const response = await fetch(`${API_BASE}${path}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(selected),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : `Assessment failed (${response.status}).`);
    return payload;
  }

  async function assessSusceptibility() {
    if (!API_BASE) { setSusceptibilityError("The hosted showcase cannot run Python inference. Start the FastAPI backend locally to calculate this result."); return; }
    setSusceptibilityBusy(true); setSusceptibilityError(""); setSusceptibility(null);
    try { setSusceptibility(await requestModel("/predict/susceptibility") as SusceptibilityPrediction); }
    catch (error) { setSusceptibilityError(error instanceof Error ? `${error.message} Check that Earth Engine is authenticated and the susceptibility API route is enabled.` : "Couldn't calculate susceptibility."); }
    finally { setSusceptibilityBusy(false); }
  }

  async function assessThreeDay() {
    if (!API_BASE) { setTemporalError("The hosted showcase cannot run Python inference. Use the saved example or run the backend locally."); return; }
    setTemporalBusy(true); setTemporalError(""); setTemporal(null); setTemporalKind(null);
    try { setTemporal(await requestModel("/predict") as TemporalPrediction); setTemporalKind("live"); }
    catch (error) { setTemporalError(error instanceof Error ? `${error.message} Check that the backend is running and CORS is configured.` : "Couldn't calculate the three-day score."); }
    finally { setTemporalBusy(false); }
  }

  function loadExample() {
    setSelected(INITIAL_POINT); setMapCenter(INITIAL_POINT); setPlaceName("Uttarkashi, Uttarakhand"); setQuery("Uttarkashi, Uttarakhand");
    setTemporal(SAVED_EXAMPLE); setTemporalKind("example"); setTemporalError(""); setSusceptibility(null); setSusceptibilityError("");
    setSearchError(""); setResults([]); setShowTemporalData(false); setShowSusceptibilityData(false);
  }

  const temporalScore = temporal?.experimental_score;
  const temporalThreshold = temporal?.threshold;
  const signal =
  typeof temporalScore === "number"
    ? getSignalBand(temporalScore)
    : null;
  const susceptibilitySignal =
  typeof susceptibility?.model_score === "number"
    ? getSignalBand(susceptibility.model_score)
    : null;
  const temporalScoreWidth = typeof temporalScore === "number" ? `${Math.max(0, Math.min(1, temporalScore)) * 100}%` : "0%";
  const susceptibilityScoreWidth = typeof susceptibility?.model_score === "number" ? `${Math.max(0, Math.min(1, susceptibility.model_score)) * 100}%` : "0%";

  return (
    <main className="app-shell" id="top" data-theme={theme}>
      <header className="topbar section-wrap">
        <a className="brand" href="#top" aria-label="PahaadWatch home"><span className="brand-icon" aria-hidden="true">⌁</span><span>pahaad<span className="brand-muted">watch</span></span></a>
        <nav className="topnav" aria-label="Main navigation"><a href="#location">Choose location</a><a href="#models">Risk checks</a><a href="#about-project">About</a></nav>
        <button className="theme-toggle" type="button" onClick={() => setTheme((value) => value === "light" ? "dark" : "light")} aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`} aria-pressed={theme === "dark"}>{theme === "light" ? "☾ Dark" : "☀ Light"}</button>
      </header>

      <section className="intro section-wrap">
        <div className="intro-copy"><span className="eyebrow"><span className="live-dot" /> FOR HIMALAYAN LOCATIONS</span>
          <h1>Check landslide<br /><em>risk around you</em></h1>
          <p>Choose a place to understand two things: Check if an area is prone to landslides and see the predicted landslide risk for the next 3 days</p>
        </div>
        <div className="intro-note"><span className="note-symbol">i</span><p><strong>What this site does</strong><br />It offers two research estimates for a selected location. They are useful for exploration—not for confirming a building site is safe or replacing official advice.</p></div>
      </section>

      <section className="location-section section-wrap" id="location">
        <div className="section-heading"><div><span className="section-kicker">STEP 1</span><h2>Choose a location</h2></div><p>Search for a town or village, use your location, or tap the map to select a specific point.</p></div>
        <div className="location-layout">
          <div className="location-panel soft-card">
            <form className="search-form" onSubmit={searchPlace}>
              <label className="sr-only" htmlFor="place-search">Search a town or village</label><span className="search-icon" aria-hidden="true">⌕</span>
              <input id="place-search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="e.g. Joshimath, Uttarakhand" maxLength={120} />
              <button type="submit" disabled={searching}>{searching ? "Searching…" : "Search"}</button>
            </form>
            <div className="location-tools"><button type="button" className="quiet-button" onClick={useMyLocation}>◎ Use my location</button><span>or tap the map</span></div>
            {searchError && <p className="message error-message" role="status">{searchError}</p>}
            {results.length > 0 && <div className="search-results" role="listbox" aria-label="Matching places">{results.map((result) => <button type="button" className="search-result" key={result.place_id} onClick={() => chooseSearchResult(result)}><span>⌖</span><span>{result.display_name}</span><span aria-hidden="true">↗</span></button>)}</div>}
            <div className="map-frame"><LocationMap center={mapCenter} selected={selected} onSelect={(point) => selectPoint(point)} /></div>
            <div className="selected-location"><span className="pin-icon">⌖</span><div><span className="section-kicker">SELECTED LOCATION</span><strong>{placeName}</strong><span className="coordinates">{selected.latitude.toFixed(5)}°, {selected.longitude.toFixed(5)}°</span></div><button type="button" className="reset-button" aria-label="Load default example location" onClick={loadExample}>↺</button></div>
          </div>
          <aside className="selection-tip"><span className="tip-icon">⌖</span><div><strong>For a property or plot</strong><p>Zoom in and tap the approximate spot. This checks one point, not the whole plot or construction safety.</p></div></aside>
        </div>
      </section>

      <section className="models-section section-wrap" id="models">
        <div className="section-heading"><div><span className="section-kicker">STEP 2</span><h2>Choose the question you want answered</h2></div><p>Run either model—or both—for the same location. Each answers a different question.</p></div>
        <div className="models-grid">
          <article
  className={`model-card soft-card susceptibility-card ${
    susceptibilitySignal
      ? `signal-${susceptibilitySignal.key}`
      : ""
  }`}
>
            <div className="model-card-top"><span className="model-icon terrain-icon">⌁</span><span className="model-type">LONGER-TERM SETTING</span></div>
            <h3>Landslide susceptibility</h3>
            <p className="model-question">“Is this place generally prone to landslides?”</p>
            <p className="model-description">Uses terrain, land cover and long-term average rainfall to estimate the location&apos;s background susceptibility.</p>
            {susceptibility ? <div className="model-result" aria-live="polite">
              <div className="result-source"><span className="source-dot" /> LIVE BACKEND RESULT</div>
              <div className="result-class-row"><span
  className={`class-chip signal-chip ${
    susceptibilitySignal
      ? `signal-${susceptibilitySignal.key}`
      : ""
  }`}
>
  {susceptibilitySignal?.label ?? "Result unavailable"}
</span><span className="score-small">Score {susceptibility.model_score.toFixed(4)}</span></div>
              <div className="score-track"><span style={{ width: susceptibilityScoreWidth }} /></div>
              <p className="result-meaning">A model-derived susceptibility class, not a probability that a landslide will occur or a judgment that a plot is safe.</p>
              <button type="button" className="details-button" aria-expanded={showSusceptibilityData} onClick={() => setShowSusceptibilityData((value) => !value)}>{showSusceptibilityData ? "Hide assessment details" : "What affects this result?"}<span>{showSusceptibilityData ? "−" : "+"}</span></button>
              {showSusceptibilityData && <div className="detail-text"><p>Inputs include elevation, slope, land-cover code, and a long-term rainfall average. The output uses the susceptibility class thresholds saved with the model.</p>{susceptibility.warning && <p>{susceptibility.warning}</p>}</div>}
            </div> : <div className="model-empty"><div className="empty-result-icon">◒</div><strong>Your susceptibility result will appear here.</strong><span>Checks the landscape&apos;s general characteristics, not the next few days.</span></div>}
            {API_BASE ? <button type="button" className="model-button" onClick={assessSusceptibility} disabled={susceptibilityBusy}>{susceptibilityBusy ? "Checking susceptibility…" : susceptibility ? "Refresh susceptibility" : "Check susceptibility"}<span>→</span></button> : <div className="hosted-note">Live model calculation needs the Python backend. Run this project locally to calculate a result.</div>}
            {susceptibilityError && <p className="message error-message" role="alert">{susceptibilityError}</p>}
          </article>

          <article
  className={`model-card soft-card temporal-card ${
    signal ? `signal-${signal.key}` : ""
  }`}
>
            <div className="model-card-top"><span className="model-icon rain-icon">≈</span><span className="model-type">SHORT-TERM OUTLOOK</span><span className="horizon-pill">NEXT 3 DAYS</span></div>
            <h3>Risk over the next 3 days</h3>
            <p className="model-question">“Check the landslide risk for the next 3 days”</p>
            <p className="model-description">Uses recent rainfall totals over 1–30 days plus terrain and land cover.</p>
            {temporal ? <div className="model-result" aria-live="polite">
              <div className="result-source"><span className="source-dot" />{temporalKind === "example" ? "SAVED EXAMPLE · NOT LIVE" : "LIVE BACKEND RESULT"}</div>
              <div className="signal-heading"><span className="signal-symbol">{signal?.key === "very-high" || signal?.key === "high" ? "!" : signal?.key === "elevated" ? "•" : "✓"}</span><div><span className="section-kicker">MODEL SIGNAL</span><div className="signal-title">{signal?.label ?? "Model result"}</div></div></div>
              <div className="score-row"><div><span className="section-kicker">MODEL SCORE · NOT A PERCENTAGE CHANCE</span><div className="score-value">{temporal.experimental_score.toFixed(4)}</div></div><span className="threshold-text">Threshold {temporal.threshold.toFixed(3)}</span></div>
              <div className="score-track"><span style={{ width: temporalScoreWidth }} /></div>
              <p className="result-meaning">{signal?.explanation}</p>
              {temporalKind === "example" && <p className="example-date">Saved Uttarkashi example · rainfall through {prettyDate(temporal.rainfall?.rainfall_data_through)}.</p>}
              <button type="button" className="details-button" aria-expanded={showTemporalData} onClick={() => setShowTemporalData((value) => !value)}>{showTemporalData ? "Hide data details" : "What data was used?"}<span>{showTemporalData ? "−" : "+"}</span></button>
              {showTemporalData && <div className="data-details">{temporal.rainfall && <div><h4>Recent rainfall</h4><div className="data-rows">{[1, 3, 7, 15, 30].map((days) => <div key={days}><span>Last {days} day{days === 1 ? "" : "s"}</span><strong>{typeof temporal.rainfall?.[`RAIN_${days}D`] === "number" ? `${Number(temporal.rainfall[`RAIN_${days}D`]).toFixed(1)} mm` : "—"}</strong></div>)}</div><p className="data-footnote">Data through {prettyDate(temporal.rainfall.rainfall_data_through)}.</p></div>}{temporal.terrain && <div><h4>Terrain</h4><div className="data-rows"><div><span>Elevation</span><strong>{Number(temporal.terrain.ELEVATION).toFixed(0)} m</strong></div><div><span>Slope</span><strong>{Number(temporal.terrain.SLOPE).toFixed(1)}°</strong></div><div><span>Land-cover code</span><strong>{temporal.terrain.LANDCOVER_CODE}</strong></div></div></div>}</div>}
            </div> : <div className="model-empty"><div className="empty-result-icon">≈</div><strong>Your 3-day result will appear here.</strong><span>Estimates an experimental signal from the recent conditions.</span></div>}
            {API_BASE ? <div className="model-actions"><button type="button" className="model-button" onClick={assessThreeDay} disabled={temporalBusy}>{temporalBusy ? "Checking 3-day outlook…" : temporal ? "Refresh 3-day outlook" : "Check next 3 days"}<span>→</span></button><button type="button" className="secondary-action" onClick={loadExample}>Load example</button></div> : <button type="button" className="model-button" onClick={loadExample}>Show saved example <span>→</span></button>}
            {temporalError && <p className="message error-message" role="alert">{temporalError}</p>}
            <p className="model-warning">Experimental research score—not an official landslide warning or calibrated probability.</p>
          </article>
        </div>
      </section>

      <section className="how-section" id="how-it-works"><div className="section-wrap how-inner"><div><span className="section-kicker">IN SIMPLE TERMS</span><h2>Two models, two different questions.</h2><p>They share a selected location, but their inputs and outputs are not interchangeable.</p></div><div className="simple-steps"><div><span className="step-icon ground">⌁</span><div><strong>Susceptibility model</strong><p>Uses relatively stable landscape features and long-term average rainfall to estimate general susceptibility.</p></div></div><div><span className="step-icon rain">≈</span><div><strong>Three-day model</strong><p>Uses recent rainfall history and terrain to estimate its experimental next-three-day target.</p></div></div><div><span className="step-icon model">i</span><div><strong>Use with care</strong><p>Neither model confirms that a property is safe or replaces local professional assessment.</p></div></div></div></div></section>

      <section className="project-details section-wrap" id="about-project"><button type="button" className="project-toggle" aria-expanded={showProjectDetails} onClick={() => setShowProjectDetails((value) => !value)}><span><span className="section-kicker">OPTIONAL DETAILS</span><strong>Data, methods and limitations</strong></span><span className="toggle-symbol">{showProjectDetails ? "−" : "+"}</span></button>{showProjectDetails && <div className="project-details-body"><div><h3>Susceptibility model</h3><p>Uses location, elevation, slope, land-cover code and a fixed long-term rainfall average. Its classes are derived from thresholds stored with the saved susceptibility model.</p></div><div><h3>Three-day model</h3><p>A Random Forest trained against a constructed historical NEXT_3D target, using 1-, 3-, 7-, 15- and 30-day IMERG rainfall totals plus elevation, slope and land cover.</p></div><div><h3>Evaluation and limits</h3><p>The IMERG model&apos;s spatial holdout test reported ROC-AUC 0.735 and PR-AUC 0.783 on a constructed test sample. Scores are uncalibrated, inventories can be incomplete and negative labels may be uncertain. Not suitable for official warnings or construction approval.</p></div><p className="tech-note">Python · scikit-learn · NASA IMERG · Google Earth Engine · FastAPI · Next.js · Leaflet</p></div>}</section>

      <footer className="footer"><div className="section-wrap footer-inner"><a className="brand" href="#top"><span className="brand-icon" aria-hidden="true">⌁</span><span>pahaad<span className="brand-muted">watch</span></span></a><span>Location-based research estimates for Himalayan locations.</span><a href="#top">Back to top ↑</a></div></footer>
    </main>
  );
}
