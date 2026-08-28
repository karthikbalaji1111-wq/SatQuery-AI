/**
 * Query entry point. The natural-language query pipeline is not implemented in
 * this foundation build; the input is intentionally disabled until the `query`
 * and `ai` services land.
 */
export function QueryPanel() {
  return (
    <section className="panel" aria-labelledby="query-heading">
      <h2 id="query-heading">Ask</h2>
      <form
        onSubmit={(event) => event.preventDefault()}
        className="query-form"
      >
        <label htmlFor="query-input">Natural-language satellite query</label>
        <textarea
          id="query-input"
          name="query"
          rows={3}
          disabled
          placeholder="e.g. Show flooding near Chennai between June and August 2024"
        />
        <button type="submit" disabled>
          Run query
        </button>
      </form>
      <p className="hint">Query execution arrives in a later phase.</p>
    </section>
  );
}
