# SQL Agent

Ask a question about your data in plain English. SQL Agent finds the right tables, writes the SQL query, runs it, and answers with the key figures, a chart and the full table.

## Databases

Pick a database in the selector at the top left. Switching starts a new conversation.

| Database | What's inside |
|---|---|
| **Chinook** | A digital music store: artists, albums, tracks, genres, playlists, customers, invoices |
| **Northwind** | A food trading company: products, categories, suppliers, customers, orders, employees, shippers |

## How it answers

1. **Understands the question.** A follow-up such as *"and in 2022?"* is rewritten into a complete question.
2. **Selects the data.** The relevant tables and columns are retrieved from the schema.
3. **Writes the query.** It plans the query, writes the SQL and checks it. Only read-only `SELECT` queries are allowed. If the database rejects the query, the agent reads the error and fixes it on its own.
4. **Shows the result.** You get a short answer, plus a result card with:
   - **Chart**: a line for trends over time, bars for rankings and breakdowns, big numbers for single values
   - **Table**: every row, sortable by clicking a column header
   - **SQL**: the exact query that was run
   - **CSV**: a download of the results

Expand **SQL Agent** above an answer to follow each step.

## Tips

- Say what you want to measure and how to group it: *"revenue per country"*, *"orders per month in 2022"*.
- Ask follow-up questions: the agent remembers the conversation.
- If the agent had to interpret a term, it says so at the end of its answer.
- At most 100 rows are read per query. The answer tells you when the results are partial.
