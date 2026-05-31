> ## Documentation Index
>
> Fetch the complete documentation index at: <https://developers.openalex.org/llms.txt>
> Use this file to discover all available pages before exploring further.

# LLM Quick Reference

> OpenAlex API reference optimized for AI agents

<Note>
  This page is optimized for LLM agents and AI applications. For human-readable guides, see [Getting Started](/quickstart).
</Note>

## Base URL and Authentication

```
Base: https://api.openalex.org
Auth: API key required (free at openalex.org/settings/api)
Rate: $1/day free usage with key, $0.01/day without
```

## Entity Endpoints

```
/works          - Hundreds of millions of scholarly documents (articles, books, datasets)
/authors        - Researcher profiles with disambiguated identities
/sources        - Journals, repositories, conferences
/institutions   - Universities, research organizations
/topics         - Subject classifications (3-level hierarchy)
/publishers     - Publishing organizations
/funders        - Funding agencies
```

## Special Endpoints

```
content.openalex.org/works/{id}.pdf - Download PDFs ($0.01 each)
/text                               - DEPRECATED, do not use
```

## Critical: Two-Step ID Lookup

<Warning>
  **Never filter by entity names directly.** Names are ambiguous. Always resolve to IDs first.
</Warning>

```bash theme={"dark"}
# WRONG - will fail or return wrong results
/works?filter=author_name:Einstein

# CORRECT - two steps
# 1. Get ID
/authors?search=Einstein
# Response: id = "A5012345678"

# 2. Filter by ID
/works?filter=authorships.author.id:A5012345678
```

This applies to: authors, institutions, sources, topics, publishers, funders.

## Query Parameters

```
api_key=        - Required (get free at openalex.org/settings/api)
filter=         - Filter results (see syntax below)
search=         - Full-text search across title/abstract/fulltext
sort=           - Sort results (e.g., cited_by_count:desc)
per_page=       - Results per page (default: 25, max: 100)
page=           - Page number for pagination
sample=         - Random results (e.g., sample=50)
seed=           - Seed for reproducible sampling
select=         - Limit returned fields (e.g., select=id,title)
group_by=       - Aggregate results by a field
```

<Note>
  OpenAlex uses **snake\_case** for all parameters: `per_page`, `group_by`, `api_key`.
</Note>

## Filter Syntax

```bash theme={"dark"}
# Single filter
?filter=publication_year:2024

# Multiple filters (AND)
?filter=publication_year:2024,is_oa:true

# Multiple values (OR) - up to 100 values
?filter=type:article|book|dataset

# Negation
?filter=type:!paratext

# Comparison
?filter=cited_by_count:>100
?filter=publication_year:<2020
?filter=publication_year:2020-2024
```

## Common Filter Fields

### Works

```
authorships.author.id         - Author's OpenAlex ID
authorships.institutions.id   - Institution's OpenAlex ID
primary_location.source.id    - Journal/source ID
topics.id                     - Topic ID
publication_year              - Year (integer)
cited_by_count                - Citations (integer)
is_oa                         - Open access (boolean)
type                          - article, book, dataset, etc.
has_fulltext                  - Has searchable fulltext (boolean)
```

### Authors

```
last_known_institutions.id    - Current institution
works_count                   - Number of works
cited_by_count                - Total citations
```

## Common Patterns

### Get works by author

```bash theme={"dark"}
# Step 1: Find author
/authors?search=Heather+Piwowar
# Step 2: Get works
/works?filter=authorships.author.id:A5023888391
```

### Get works from institution

```bash theme={"dark"}
# Step 1: Find institution
/institutions?search=MIT
# Step 2: Get works
/works?filter=authorships.institutions.id:I63966007
```

### Bulk DOI lookup (up to 100)

```bash theme={"dark"}
/works?filter=doi:10.1234/a|10.1234/b|10.1234/c&per_page=100
```

### Random sample

```bash theme={"dark"}
/works?sample=100&seed=42
```

### Aggregate by field

```bash theme={"dark"}
/works?filter=publication_year:2024&group_by=topics.id
```

## Pricing

| Endpoint                   | Cost     |
| -------------------------- | -------- |
| Singleton (`/works/W123`)  | Free     |
| List (`/works?filter=...`) | \$0.0001 |
| Search (`?search=`)        | \$0.001  |
| Content download (PDF)     | \$0.01   |

## Query Limits

| Limit                | Value          |
| -------------------- | -------------- |
| OR values per filter | 100            |
| `per_page` max       | 100            |
| `sample` max         | 10,000         |
| Basic paging limit   | 10,000 results |

## Error Handling

```python theme={"dark"}
def fetch_with_retry(url, max_retries=5):
    for attempt in range(max_retries):
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.json()
        if response.status_code in [429, 500]:
            time.sleep(2 ** attempt)  # Exponential backoff
            continue
        response.raise_for_status()
    raise Exception("Max retries exceeded")
```

## Common Mistakes

| Mistake               | Fix                             |
| --------------------- | ------------------------------- |
| Filter by name        | Resolve to ID first             |
| Default page size     | Use `per_page=100`              |
| Sequential ID lookups | Batch with `\|` operator        |
| No error handling     | Implement exponential backoff   |
| Fetching all fields   | Use `select=` for needed fields |

## Deprecated Features

See [Deprecations](/guides/deprecations) for full list. Key items:

* **Concepts** → Use Topics instead
* **`/text` endpoint** → Do not use
* **`host_venue`** → Use `primary_location`
* **`grants`** → Use `funders` and `awards`

## Links

* Full docs: [https://docs.openalex.org](https://docs.openalex.org)
* API key: [https://openalex.org/settings/api](https://openalex.org/settings/api)
* Help: [https://openalex.org/help](https://openalex.org/help)
